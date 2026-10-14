"""Full grounding-connector training on COCO train2017 (118k images), vision tower run live.

Same connector / prompts / targets as train_connector.py; the frozen Qwen3.8 vision tower runs on
the GPU in bf16 each step (caching 118k images of merger inputs would need ~150 GB).
Resumable: checkpoint (connector + optimizer + step) every 500 steps in analysis/grounding/<tag>/.
Evaluate afterwards with: train_connector.py eval --tag <tag>

usage: train_full.py [--steps 14800] [--lr 2e-4] [--init run1] [--tag run_full]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel

import train_connector as T

ROOT = T.ROOT
FULL = ROOT / "analysis/grounding_full"
IMGS = ROOT / "coco/train448"


class Coco(Dataset):
    def __init__(self):
        z = np.load(FULL / "train_meta.npz")
        self.stems, self.W, self.H = z["stems"], z["W"], z["H"]
        self.obj_off, self.objs, self.cap_off = z["obj_off"], z["objs"], z["cap_off"]
        self.caps = json.load(open(FULL / "train_caps.json"))
        self.names = json.load(open(FULL / "cats.json"))
        self.proc = None

    def __len__(self):
        return len(self.stems)

    def meta(self, i):
        o = self.objs[self.obj_off[i]:self.obj_off[i + 1]]
        return dict(W=int(self.W[i]), H=int(self.H[i]),
                    objs=[dict(name=self.names[int(r[0])], box=r[1:5].tolist(), area=float(r[5])) for r in o],
                    caps=self.caps[self.cap_off[i]:self.cap_off[i + 1]])

    def __getitem__(self, i):
        if self.proc is None:  # one processor per worker
            self.proc = AutoImageProcessor.from_pretrained(ROOT / "dl/qwen38fn_cfg")
            self.proc.size = {"longest_edge": 4096 * 4096, "shortest_edge": 32 * 32}
        try:
            img = Image.open(IMGS / f"{self.stems[i]}.jpg").convert("RGB")
        except OSError:
            return None
        inp = self.proc(images=[img], return_tensors="pt")
        rng = random.Random(i * 7919 + int(time.time() * 1000) % 100000)
        q, a = T.make_example(self.meta(i), rng)
        return inp["pixel_values"], inp["image_grid_thw"], q, a


class Offset(Sampler):
    """Fixed random order (seeded) over the given indices, starting at a sample index (for resume)."""
    def __init__(self, idx, start, seed=0):
        self.order = [idx[j] for j in np.random.default_rng(seed).permutation(len(idx))]
        self.start = start % len(idx)

    def __iter__(self):
        yield from self.order[self.start:]
        while True:  # wrap around for extra epochs
            yield from self.order

    def __len__(self):
        return 10 ** 9


def collate(b):
    return [x for x in b if x is not None]


def load_vision():
    cfg = json.load(open(ROOT / "dl/qwen38fn_cfg/config.json"))["vision_config"]
    cfg.pop("model_type", None)
    with torch.device("cuda"):
        vm = Qwen3_5VisionModel(Qwen3_5VisionConfig(**cfg))
    own = vm.state_dict()
    with torch.no_grad(), safe_open(T.QWEN_SHARD, framework="pt", device="cuda") as f:
        for k in f.keys():
            if k.startswith("model.visual.") and k[13:] in own:
                own[k[13:]].copy_(f.get_tensor(k))
    vm = vm.to(torch.bfloat16).eval()
    for p in vm.parameters():
        p.requires_grad_(False)
    grab = {}
    vm.merger.linear_fc1.register_forward_pre_hook(lambda m, a: grab.__setitem__("x", a[0].detach()))
    return vm, grab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=14800)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init", default="run1")
    p.add_argument("--tag", default="run_full")
    a = p.parse_args()
    out = T.DATA / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt.pt"

    # data first: DataLoader workers fork now, from a small process (the RAM cap counts them too)
    step = json.load(open(out / "ckpt_step.json"))["step"] if (out / "ckpt_step.json").exists() and ck.exists() else 0
    ds = Coco()
    present = {f.stem for f in IMGS.glob("*.jpg")}
    idx = [i for i, s_ in enumerate(ds.stems) if s_ in present]
    print(f"{len(idx)}/{len(ds)} images available", flush=True)
    per_step = a.bs * a.accum
    dl = DataLoader(ds, batch_size=a.bs, sampler=Offset(idx, step * per_step), num_workers=0,
                    collate_fn=collate)  # no worker process: a forked worker copies ~0.5 GB
    it = iter(dl)
    next(it)  # start workers

    tok, llm = T.load_llm("cuda")
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    vm, grab = load_vision()
    conn = T.Connector(llm.config.hidden_size).cuda()
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    if ck.exists():
        s = torch.load(ck, map_location="cuda")
        conn.load_state_dict(s["conn"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    else:
        conn.load_state_dict(torch.load(T.DATA / a.init / "connector.pt", map_location="cuda"))
        print(f"initialised from {a.init}", flush=True)

    B = T.Batcher(tok, None, None, None)
    sched = lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    t0, run, t_last = time.time(), [], time.time()
    start = step
    while step < a.steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            batch = next(it)
            while not batch:
                batch = next(it)
            pv = torch.cat([b[0] for b in batch]).to("cuda", torch.bfloat16)
            thw = torch.cat([b[1] for b in batch]).to("cuda")
            with torch.no_grad():
                vm(pv, grid_thw=thw)
            xs = grab["x"].float().split([int(t * h * w) // 4 for t, h, w in thw.tolist()])
            items = []
            for (_, _, q, ans), x in zip(batch, xs):
                pre, post = B.prompt_parts(q)
                items.append((pre, x, post, tok(ans, add_special_tokens=False).input_ids + [B.eot]))
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
            el = time.time() - t0
            rate = el / (step - start)
            print(f"step {step}/{a.steps}  loss {np.mean(run[-50:]):.3f}  {rate:.2f}s/step  "
                  f"eta {(a.steps - step) * rate / 3600:.1f}h  vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % 200 == 0 or step == a.steps:
            tmp = out / "ckpt.tmp"
            torch.save(dict(conn=conn.state_dict(), opt=opt.state_dict(), step=step), tmp)
            tmp.rename(ck)
            json.dump(dict(step=step), open(out / "ckpt_step.json", "w"))
            torch.save(conn.state_dict(), out / "connector.pt")
            json.dump(dict(loss_tail=run[-500:], step=step), open(out / "train_log.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
