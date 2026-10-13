"""Train a grounding connector: Qwen3.8 vision features -> frozen Llama-3.2-1B.

The connector has exactly the shape of Qwen's merger (mm.0: 4608->4608, GELU(tanh), mm.2: 4608->2048),
so it exports straight into the mmproj and runs in llama.cpp unchanged. Llama stays frozen; only the
connector is trained, with next-token loss on answers that contain boxes and captions.

Box format: [x1,y1,x2,y2] in 0..999, relative to the image.

usage: train_connector.py prep            # pack cached shards into one memmap + build splits
       train_connector.py train [--steps N] [--tag T]
       train_connector.py eval  [--tag T] [--n 300]
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis/merger_in"
DATA = ROOT / "analysis/grounding"
LLAMA = ROOT / "dl/llama32-1b-hf"
QWEN_SHARD = ROOT / "dl/qwen38fn-00001.safetensors"
TESTS = ["000000039769", "000000000139", "000000000785", "000000000632", "000000001000", "000000000724"]
IMG_BEG, IMG_END, MARKER = "<|vision_start|>", "<|vision_end|>", "<__media__>"
Q_LIST = "List the objects in this image with their bounding boxes."
Q_DESC = "Describe this image in one sentence."
Q_WHERE = "Where is the {}? Answer with bounding boxes."


# ---------------------------------------------------------------- data prep
def prep():
    DATA.mkdir(parents=True, exist_ok=True)
    shards = sorted(CACHE.glob("shard_*.npz"))
    metas, n = [], 0
    for f in shards:
        z = np.load(f)
        for s, gh, gw, W, H, a, b in zip(z["stems"], z["gh"], z["gw"], z["W"], z["H"], z["offs"][:-1], z["offs"][1:]):
            metas.append(dict(stem=str(s), gh=int(gh), gw=int(gw), W=int(W), H=int(H), off=n + int(a), n=int(b - a)))
        n += int(z["offs"][-1])
    X = np.lib.format.open_memmap(DATA / "x.npy", mode="w+", dtype=np.float16, shape=(n, 4608))
    pos = 0
    for f in shards:
        x = np.load(f)["x"]
        X[pos:pos + len(x)] = x
        pos += len(x)
        del x
    X.flush()
    inst = json.load(open(ROOT / "coco/annotations/instances_val2017.json"))
    caps = json.load(open(ROOT / "coco/annotations/captions_val2017.json"))
    cname = {c["id"]: c["name"] for c in inst["categories"]}
    boxes, capd = {}, {}
    for a in inst["annotations"]:
        boxes.setdefault(a["image_id"], []).append((cname[a["category_id"]], a["bbox"], a["area"], a["iscrowd"]))
    for a in caps["annotations"]:
        capd.setdefault(a["image_id"], []).append(a["caption"].strip())
    for m in metas:
        i = int(m["stem"])
        m["objs"] = [dict(name=c, box=b, area=ar) for c, b, ar, crowd in boxes.get(i, []) if not crowd]
        m["caps"] = capd.get(i, [])
    rng = random.Random(0)
    stems = [m["stem"] for m in metas if m["stem"] not in TESTS]
    rng.shuffle(stems)
    split = dict(eval=TESTS + stems[:494], train=stems[494:])
    json.dump(dict(metas=metas, split=split), open(DATA / "meta.json", "w"))
    print(f"{len(metas)} images, {n} tokens, train {len(split['train'])}, eval {len(split['eval'])}")


def box_str(b, W, H):
    x, y, w, h = b
    q = lambda v, d: max(0, min(999, int(round(v / d * 999))))
    return f"[{q(x, W)},{q(y, H)},{q(x + w, W)},{q(y + h, H)}]"


def make_example(m, rng):
    """(question, answer) for one image; random task."""
    W, H = m["W"], m["H"]
    objs = [o for o in m["objs"] if o["area"] >= 0.004 * W * H]
    objs.sort(key=lambda o: -o["area"])
    r = rng.random()
    if objs and r < 0.45:
        lines = [f"{o['name']} {box_str(o['box'], W, H)}" for o in objs[:12]]
        return Q_LIST, "\n".join(lines)
    if objs and r < 0.8:
        name = rng.choice(sorted({o["name"] for o in objs}))
        bs = [box_str(o["box"], W, H) for o in objs if o["name"] == name][:6]
        return Q_WHERE.format(name), " ".join(bs)
    return Q_DESC, rng.choice(m["caps"]) if m["caps"] else "An image."


# ---------------------------------------------------------------- model
class Connector(nn.Module):
    def __init__(self, d_out=2048):
        super().__init__()
        self.fc1 = nn.Linear(4608, 4608)
        self.fc2 = nn.Linear(4608, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def init_connector(conn, emb_norm, sample):
    from safetensors import safe_open
    with safe_open(QWEN_SHARD, framework="pt") as f:
        conn.fc1.weight.data.copy_(f.get_tensor("model.visual.merger.linear_fc1.weight").float())
        conn.fc1.bias.data.copy_(f.get_tensor("model.visual.merger.linear_fc1.bias").float())
    nn.init.normal_(conn.fc2.weight, std=0.02)
    nn.init.zeros_(conn.fc2.bias)
    with torch.no_grad():  # scale so outputs start near the size of Llama's word embeddings
        out = conn(sample).norm(dim=-1).mean()  # real merger inputs, not noise
        conn.fc2.weight.mul_(emb_norm / out)


class Batcher:
    def __init__(self, tok, X, metas, rng):
        self.tok, self.X, self.metas, self.rng = tok, X, metas, rng
        self.eot = tok.convert_tokens_to_ids("<|eot_id|>")

    def prompt_parts(self, question):
        msgs = [{"role": "user", "content": MARKER + question}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pre, post = text.split(MARKER)
        pre_ids = self.tok(pre + IMG_BEG, add_special_tokens=False).input_ids
        post_ids = self.tok(IMG_END + post, add_special_tokens=False).input_ids
        return pre_ids, post_ids

    def example(self, m, question, answer):
        pre, post = self.prompt_parts(question)
        ans = self.tok(answer, add_special_tokens=False).input_ids + [self.eot]
        x = torch.from_numpy(np.asarray(self.X[m["off"]:m["off"] + m["n"]], dtype=np.float32))
        return pre, x, post, ans


def embed_batch(llm, conn, items, device):
    """items: list of (pre_ids, img_feats, post_ids, ans_ids). Returns inputs_embeds, attn, labels."""
    E = llm.get_input_embeddings()
    seqs, labs = [], []
    for pre, x, post, ans in items:
        img = conn(x.to(device))
        tok = lambda ids: E(torch.tensor(ids, device=device))
        emb = torch.cat([tok(pre), img.to(E.weight.dtype), tok(post + ans)])
        lab = torch.full((len(emb),), -100, dtype=torch.long, device=device)
        if ans:
            lab[-len(ans):] = torch.tensor(ans, device=device)
        seqs.append(emb); labs.append(lab)
    L = max(len(s) for s in seqs)
    d = seqs[0].shape[1]
    out = torch.zeros(len(seqs), L, d, dtype=seqs[0].dtype, device=device)
    att = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long, device=device)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        out[i, :len(s)] = s; att[i, :len(s)] = 1; lab[i, :len(s)] = l
    return out, att, lab


def load_llm(device):
    from transformers import AutoTokenizer, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained(LLAMA)
    with torch.device(device):
        llm = LlamaForCausalLM.from_pretrained(LLAMA, dtype=torch.bfloat16, device_map=device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    return tok, llm


# ---------------------------------------------------------------- train
def train(a):
    device = "cuda"
    meta = json.load(open(DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    train_m = [by[s] for s in meta["split"]["train"]]
    X = np.load(DATA / "x.npy", mmap_mode="r")
    tok, llm = load_llm(device)
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    emb_norm = llm.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
    conn = Connector(llm.config.hidden_size)
    init_connector(conn, emb_norm, torch.from_numpy(np.asarray(X[::max(1, len(X) // 512)][:512], dtype=np.float32)))
    conn.to(device)
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    rng = random.Random(1)
    B = Batcher(tok, X, train_m, rng)
    sched = lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    t0, run = time.time(), []
    out_dir = DATA / a.tag
    out_dir.mkdir(exist_ok=True)
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            items = []
            for m in rng.sample(train_m, a.bs):
                q, ans = make_example(m, rng)
                items.append(B.example(m, q, ans))
            emb, att, lab = embed_batch(llm, conn, items, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hid = llm.model(inputs_embeds=emb, attention_mask=att).last_hidden_state
                tgt = lab[:, 1:]
                sel = tgt != -100  # vocab projection only where there is a target: saves ~0.6 GB
                logits = llm.lm_head(hid[:, :-1][sel]).float()
                loss = F.cross_entropy(logits, tgt[sel]) / a.accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(conn.parameters(), 1.0)
        opt.step()
        run.append(tot)
        if step % 25 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.steps}  loss {np.mean(run[-25:]):.3f}  {el / step:.2f}s/step  "
                  f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % 500 == 0 or step == a.steps:
            torch.save(conn.state_dict(), out_dir / "connector.pt")
    json.dump(dict(loss=run, args=vars(a)), open(out_dir / "train_log.json", "w"))
    print("saved", out_dir / "connector.pt")


# ---------------------------------------------------------------- eval
def parse_boxes(text):
    return [tuple(int(v) for v in b) for b in re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def evaluate(a):
    device = "cuda"
    meta = json.load(open(DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    ev = [by[s] for s in meta["split"]["eval"]][: a.n]
    train_m = [by[s] for s in meta["split"]["train"]]
    X = np.load(DATA / "x.npy", mmap_mode="r")
    tok, llm = load_llm(device)
    conn = Connector(llm.config.hidden_size).to(device)
    conn.load_state_dict(torch.load(DATA / a.tag / "connector.pt"))
    B = Batcher(tok, X, ev, random.Random(0))
    # prior baseline: average box of each category over the training set (no image needed)
    prior = {}
    for m in train_m:
        for o in m["objs"]:
            prior.setdefault(o["name"], []).append(parse_boxes(box_str(o["box"], m["W"], m["H"]))[0])
    prior = {k: tuple(np.mean(v, 0)) for k, v in prior.items()}

    def gen(m, q, n=160):
        pre, x, post, _ = B.example(m, q, "")
        emb, att, _ = embed_batch(llm, conn, [(pre, x, post, [])], device)
        out = llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    rows, where_iou, prior_iou, center_iou, prec, rec, examples = [], [], [], [], [], [], {}
    for k, m in enumerate(ev):
        W, H = m["W"], m["H"]
        objs = [o for o in m["objs"] if o["area"] >= 0.004 * W * H]
        if not objs:
            continue
        gt = {}
        for o in objs:
            gt.setdefault(o["name"], []).append(parse_boxes(box_str(o["box"], W, H))[0])
        name = max(gt, key=lambda n: max((b[2] - b[0]) * (b[3] - b[1]) for b in gt[n]))
        ans = gen(m, Q_WHERE.format(name), 60)
        pb = parse_boxes(ans)
        best = max((iou(p, g) for p in pb[:1] for g in gt[name]), default=0.0)
        where_iou.append(best)
        prior_iou.append(max(iou(prior.get(name, (250, 250, 750, 750)), g) for g in gt[name]))
        center_iou.append(max(iou((250, 250, 750, 750), g) for g in gt[name]))
        lst = gen(m, Q_LIST)
        said = {re.sub(r"\s*\[.*", "", ln).strip().lower() for ln in lst.splitlines() if "[" in ln}
        prec.append(len(said & set(gt)) / max(1, len(said)))
        rec.append(len(said & set(gt)) / len(gt))
        if m["stem"] in TESTS:
            examples[m["stem"]] = dict(where_q=name, where=ans, list=lst, desc=gen(m, Q_DESC, 60), gt={k: v for k, v in gt.items()})
        if (k + 1) % 50 == 0:
            print(f"{k + 1}: where IoU {np.mean(where_iou):.3f} acc@0.5 {np.mean(np.array(where_iou) >= .5):.3f} | "
                  f"prior {np.mean(prior_iou):.3f} | list P {np.mean(prec):.3f} R {np.mean(rec):.3f}", flush=True)
    res = dict(n=len(where_iou), where_iou=float(np.mean(where_iou)), where_acc50=float(np.mean(np.array(where_iou) >= .5)),
               prior_iou=float(np.mean(prior_iou)), prior_acc50=float(np.mean(np.array(prior_iou) >= .5)),
               center_iou=float(np.mean(center_iou)), list_precision=float(np.mean(prec)), list_recall=float(np.mean(rec)),
               examples=examples)
    json.dump(res, open(DATA / a.tag / "eval.json", "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in res.items() if k != "examples"})
    for s, e in examples.items():
        print(f"--- {s}  where is the {e['where_q']}? -> {e['where']}   (gt {e['gt'][e['where_q']]})\n"
              f"    list: {e['list'][:200]!r}\n    desc: {e['desc']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "train", "eval"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tag", default="run1")
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()
    {"prep": prep, "train": train, "eval": evaluate}[a.cmd](a) if a.cmd != "prep" else prep()
