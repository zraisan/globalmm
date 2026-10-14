"""Hybrid connector: each image patch reaches Llama as TWO inputs, interleaved in raster order:
  1. our trained connector's token (Qwen3.8 vision features; carries *where*)
  2. a decoded-word embedding: Qwen3-VL-2B's first LLM layer decoded to its top-5 words, re-embedded
     with Llama's own (frozen) word embeddings (carries open-vocabulary *what*; from prep_hybrid.py)
Only the connector trains. --no-words trains the same connector the same way without (2): the control.

usage: train_hybrid.py --tag hybrid [--steps 3000] [--no-words]
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_connector as T
import train_full as TF

HYB = T.ROOT / "analysis/hybrid"


class Decode:
    """decoded-word cache -> per-token Llama word vectors"""
    def __init__(self, split):
        z = np.load(HYB / f"decode_{split}.npz")
        self.row = {s: i for i, s in enumerate(z["stems"].tolist())}
        self.off, self.ids, self.p = z["off"], z["ids"], z["p"]

    def vectors(self, stem, Emb):
        i = self.row[stem]
        ids = torch.from_numpy(self.ids[self.off[i]:self.off[i + 1]].astype(np.int64)).to(Emb.device)
        p = torch.from_numpy(self.p[self.off[i]:self.off[i + 1]].astype(np.float32)).to(Emb.device)
        p = torch.where(ids >= 0, p, torch.zeros_like(p))
        return (Emb[ids.clamp_min(0)].float() * p[..., None]).sum(1)   # [n, d]


class IdxCoco(TF.Coco):
    """same samples as the full run, plus the image index (for the decoded-word lookup)"""
    def __getitem__(self, i):
        return i, super().__getitem__(i)


def embed_hybrid(llm, conn, items, words=True):
    """items: (pre_ids, x, word_vecs, post_ids, ans_ids) -> inputs_embeds, attention, labels"""
    E = llm.get_input_embeddings()
    seqs, labs = [], []
    for pre, x, wv, post, ans in items:
        c = conn(x).to(E.weight.dtype)
        img = torch.stack([c, wv.to(c.dtype)], 1).reshape(-1, c.shape[-1]) if words else c
        tok = lambda ids: E(torch.tensor(ids, device=c.device))
        emb = torch.cat([tok(pre), img, tok(post + ans)])
        lab = torch.full((len(emb),), -100, dtype=torch.long, device=c.device)
        if ans:
            lab[-len(ans):] = torch.tensor(ans, device=c.device)
        seqs.append(emb); labs.append(lab)
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, seqs[0].shape[1], dtype=seqs[0].dtype, device=seqs[0].device)
    att = torch.zeros(len(seqs), L, dtype=torch.long, device=out.device)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long, device=out.device)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        out[i, :len(s)] = s; att[i, :len(s)] = 1; lab[i, :len(s)] = l
    return out, att, lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tag", default="hybrid")
    p.add_argument("--no-words", action="store_true")
    p.add_argument("--data", choices=["subset", "full"], default="subset")
    p.add_argument("--init", choices=["run_full", "qwen"], default="run_full",
                   help="run_full: continue the full connector; qwen: fresh, fc1 from Qwen's merger (as run1)")
    a = p.parse_args()
    out = T.DATA / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt.pt"
    step = torch.load(ck, map_location="cpu")["step"] if ck.exists() else 0

    dec = None if a.no_words else Decode("train")
    ds = IdxCoco()
    if a.data == "full":
        present = {f.stem for f in (T.ROOT / "coco/train448").glob("*.jpg")}
        idx = [i for i, s in enumerate(ds.stems.tolist()) if s in present]
    else:
        sub = Decode("train").row if dec is None else dec.row
        idx = [i for i, s in enumerate(ds.stems.tolist()) if s in sub]
    print(f"{len(idx)} training images", flush=True)
    per_step = a.bs * a.accum
    dl = DataLoader(ds, batch_size=a.bs, sampler=TF.Offset(idx, step * per_step, seed=1), num_workers=0,
                    collate_fn=lambda b: [(i, x) for i, x in b if x is not None])
    it = iter(dl)

    tok, llm = T.load_llm("cuda")
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    Emb = llm.get_input_embeddings().weight
    vm, grab = TF.load_vision()
    conn = T.Connector(llm.config.hidden_size).cuda()
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    if ck.exists():
        s = torch.load(ck, map_location="cuda")
        conn.load_state_dict(s["conn"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    elif a.init == "run_full":
        conn.load_state_dict(torch.load(T.DATA / "run_full/connector.pt", map_location="cuda"))
        print("initialised from run_full", flush=True)
    else:  # matched-budget baseline: same initialisation as run1, measured on real merger inputs
        b = next(it)
        while not b:
            b = next(it)
        with torch.no_grad():
            vm(torch.cat([x[1][0] for x in b]).to("cuda", torch.bfloat16), grid_thw=torch.cat([x[1][1] for x in b]).to("cuda"))
        emb_norm = llm.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
        conn.cpu(); T.init_connector(conn, emb_norm, grab["x"].float().cpu()[:512]); conn.cuda()
        opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
        print("fresh connector (Qwen merger fc1 init)", flush=True)
    B = T.Batcher(tok, None, None, None)
    sched = lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    t0, run, start, skipped = time.time(), [], step, 0
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
            pv = torch.cat([b[1][0] for b in batch]).to("cuda", torch.bfloat16)
            thw = torch.cat([b[1][1] for b in batch]).to("cuda")
            with torch.no_grad():
                vm(pv, grid_thw=thw)
            xs = grab["x"].float().split([int(t * h * w) // 4 for t, h, w in thw.tolist()])
            items = []
            for (i, (_, _, q, ans)), x in zip(batch, xs):
                if dec is not None:
                    with torch.no_grad():
                        wv = dec.vectors(ds.stems[i], Emb)
                    if len(wv) != len(x):
                        skipped += 1
                        continue
                else:
                    wv = None
                pre, post = B.prompt_parts(q)
                items.append((pre, x, wv, post, tok(ans, add_special_tokens=False).input_ids + [B.eot]))
            if not items:
                continue
            emb, att, lab = embed_hybrid(llm, conn, items, words=not a.no_words)
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
                  f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G  skipped {skipped}", flush=True)
        if step % 200 == 0 or step == a.steps:
            tmp = out / "ckpt.tmp"
            torch.save(dict(conn=conn.state_dict(), opt=opt.state_dict(), step=step), tmp)
            tmp.rename(ck)
            torch.save(conn.state_dict(), out / "connector.pt")
            json.dump(dict(step=step, loss_tail=run[-500:], args=vars(a)), open(out / "train_log.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
