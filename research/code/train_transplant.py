"""Trained transplant: copy a real VLM's vision tower AND its first k LLM layers (frozen), then learn only the
transformation into the target LLM.

    image -> Qwen3-VL-2B vision tower + DeepStack + LLM layers 0..k-1   (copied, frozen)
          -> RMS-normalise -> connector: LayerNorm, Linear 2048->4096, GELU, Linear 4096->2048   (trained)
          -> frozen Llama-3.2-1B
One token per 32 px patch, so boxes stay possible. k = 0 is the no-transplant baseline (same tower).
Same data and schedule for every k: 32k train2017 images (the decode-cache subset), COCO grounding tasks.

usage: train_transplant.py train --k 4 [--steps 3000]
       train_transplant.py eval  --k 4            (COCO 192 + non-COCO LVIS 246, same scoring as before)
-> analysis/grounding/transplant_k<k>/
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import eval_extra as E
import eval_ood as O
import train_connector as T
import train_full as TF

MD = T.ROOT / "dl/qwen3vl-2b"


class Stop(Exception):
    pass


def fit32(img, side=448):
    s = side / max(img.size)
    return img.resize(tuple(max(32, round(v * s / 32) * 32) for v in img.size), Image.BICUBIC)


class Donor:
    """Qwen3-VL-2B cut after k LLM layers; returns the image tokens' hidden states [n, 2048]."""
    def __init__(self, k):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(MD)
        self.proc.image_processor.size = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}
        m = Qwen3VLForConditionalGeneration.from_pretrained(MD, dtype=torch.bfloat16, device_map="cuda").eval()
        lm = m.model.language_model
        lm.layers = lm.layers[:k]            # drop every layer after the transplant
        for p in m.parameters():
            p.requires_grad_(False)
        torch.cuda.empty_cache()
        self.cap = {}

        def stop(mod, args):
            self.cap["h"] = args[0].detach()
            raise Stop
        lm.norm.register_forward_pre_hook(stop)   # the residual stream after the last kept layer
        self.m, self.img_tok = m, m.config.image_token_id
        self.text = self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}], add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, img):
        inp = self.proc(text=[self.text], images=[img], return_tensors="pt").to("cuda")
        try:
            self.m(**inp)
        except Stop:
            pass
        pos = (inp["input_ids"][0] == self.img_tok).nonzero().squeeze(1)
        h = self.cap["h"][0, pos].float()
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


class TConn(nn.Module):
    def __init__(self, d=2048, hid=4096, out=2048):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln(x)), approximate="tanh"))


def train(a, out):
    ck = out / "ckpt.pt"
    donor = Donor(a.k)
    tok, llm = T.load_llm("cuda")
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    ds = TF.Coco()
    if a.data == "full":  # every resized train2017 image (one epoch = 14,800 steps at batch 8)
        present = {f.stem for f in (T.ROOT / "coco/train448").glob("*.jpg")}
        stems = [s for s in ds.stems.tolist() if s in present]
    else:                 # the 32k decode-cache subset
        stems = np.load(T.ROOT / "analysis/hybrid/decode_train.npz")["stems"].tolist()
    row = {s: i for i, s in enumerate(ds.stems.tolist())}
    idx = [row[s] for s in stems]
    print(f"{len(idx)} training images", flush=True)
    conn = TConn().cuda()
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    step = 0
    if ck.exists():
        s = torch.load(ck, map_location="cuda")
        conn.load_state_dict(s["conn"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    else:  # start with outputs at the size of Llama's word embeddings
        with torch.no_grad():
            x = torch.cat([donor(Image.open(T.ROOT / "coco/train448" / f"{stems[i]}.jpg").convert("RGB")) for i in range(8)])
            emb = llm.get_input_embeddings().weight.float().norm(dim=-1).mean()
            conn.fc2.weight.mul_(emb / conn(x).norm(dim=-1).mean()); conn.fc2.bias.zero_()
        print(f"fresh connector, k={a.k}", flush=True)
    order = [idx[j] for j in np.random.default_rng(7).permutation(len(idx))]
    ptr = step * a.bs * a.accum
    rng = random.Random(1000 + step)
    B = T.Batcher(tok, None, None, None)
    sched = lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    t0, run, start = time.time(), [], step
    while step < a.steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            items = []
            while len(items) < a.bs:
                i = order[ptr % len(order)]; ptr += 1
                try:
                    img = Image.open(T.ROOT / "coco/train448" / f"{ds.stems[i]}.jpg").convert("RGB")
                except OSError:
                    continue
                q, ans = T.make_example(ds.meta(i), rng)
                pre, post = B.prompt_parts(q)
                items.append((pre, donor(img), post, tok(ans, add_special_tokens=False).input_ids + [B.eot]))
            emb, att, lab = T.embed_batch(llm, conn, items, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hid = llm.model(inputs_embeds=emb, attention_mask=att).last_hidden_state
                tgt = lab[:, 1:]
                sel = tgt != -100
                loss = F.cross_entropy(llm.lm_head(hid[:, :-1][sel]).float(), tgt[sel]) / a.accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(conn.parameters(), 1.0)
        opt.step()
        run.append(tot)
        if step % 50 == 0:
            rate = (time.time() - t0) / (step - start)
            print(f"step {step}/{a.steps}  loss {np.mean(run[-50:]):.3f}  {rate:.2f}s/step  eta {(a.steps - step) * rate / 3600:.1f}h  "
                  f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % 200 == 0 or step == a.steps:
            tmp = out / "ckpt.tmp"
            torch.save(dict(conn=conn.state_dict(), opt=opt.state_dict(), step=step), tmp)
            tmp.rename(ck)
            torch.save(conn.state_dict(), out / "connector.pt")
            json.dump(dict(step=step, loss_tail=run[-500:], args=vars(a)), open(out / "train_log.json", "w"))
    print("DONE", flush=True)


@torch.no_grad()
def evaluate(a, out):
    donor = Donor(a.k)
    tok, llm = T.load_llm("cuda")
    conn = TConn().cuda()
    conn.load_state_dict(torch.load(out / "connector.pt"))
    meta = json.load(open(T.DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    coco = [m for m in [by[s] for s in meta["split"]["eval"]][:200] if E.gt_of(m)]
    lvis = json.load(open(T.DATA / "lvis_ood.json"))
    B = T.Batcher(tok, None, None, None)
    feats = {}

    def gen(stem, q, n):
        if stem not in feats:
            feats.clear()
            feats[stem] = donor(fit32(Image.open(T.ROOT / "coco/val2017" / f"{stem}.jpg").convert("RGB")))
        pre, post = B.prompt_parts(q)
        emb, att, _ = T.embed_batch(llm, conn, [(pre, feats[stem], post, [])], "cuda")
        o = llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0], skip_special_tokens=True).strip()

    rows = []
    for m in coco:
        gt = E.gt_of(m)
        name = max(gt, key=lambda n: max(x for _, x in gt[n]))
        w = gen(m["stem"], T.Q_WHERE.format(name), 60)
        lst = gen(m["stem"], T.Q_LIST, 160)
        rows.append(dict(stem=m["stem"], name=name, where=w, where_iou=E.score_where(w, gt, name), list=lst, **E.score_list(lst, gt, 160)))
    ious = np.array([r["where_iou"] for r in rows])
    cs = dict(n=len(rows), where_iou=float(ious.mean()), where_acc50=float((ious >= .5).mean()),
              **{k: float(np.mean([r[k] for r in rows])) for k in ("cls_p", "cls_r", "box_p", "box_r", "dup")})
    orows = []
    for r in lvis:
        ans = gen(r["stem"], T.Q_WHERE.format(r["name"]), 60)
        pb = T.parse_boxes(ans)
        orows.append(dict(stem=r["stem"], name=r["name"], size=r["size"], answer=ans, no_box=not pb,
                          iou=max(T.iou(pb[0], tuple(g)) for g in r["boxes"]) if pb else 0.0))
    osum = O.summarize(orows)
    print(f"k={a.k} COCO", json.dumps(cs), "| LVIS", json.dumps(osum), flush=True)
    json.dump(dict(k=a.k, coco=cs, lvis=osum, coco_rows=rows, lvis_rows=orows), open(out / "eval.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tag", default=None)
    p.add_argument("--data", choices=["subset", "full"], default="subset")
    a = p.parse_args()
    out = T.DATA / (a.tag or f"transplant_k{a.k}")
    out.mkdir(parents=True, exist_ok=True)
    {"train": train, "eval": evaluate}[a.cmd](a, out)
