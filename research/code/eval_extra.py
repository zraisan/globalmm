"""Extended grounding evaluation for the write-up (same 192 held-out val2017 images as train_connector eval).

Conditions per connector tag:
  image     - the image's own Qwen3.8 features (normal use)
  shuffled  - features of a different eval image (control: what the language prior alone produces)
Metrics: where-IoU / Acc@0.5 overall and by object size; list class-level P/R, box-level P/R @0.5,
duplicate-line rate, truncation rate. Raw generations are saved.

usage: eval_extra.py TAG [TAG...]        -> analysis/grounding/<TAG>/eval_extra.json
"""
import json
import random
import re
import sys

import numpy as np
import torch

import train_connector as T

SIZES = [("small", 0, 0.01), ("medium", 0.01, 0.10), ("large", 0.10, 1.01)]


def gt_of(m):
    W, H = m["W"], m["H"]
    objs = [o for o in m["objs"] if o["area"] >= 0.004 * W * H]
    gt = {}
    for o in objs:
        gt.setdefault(o["name"], []).append((T.parse_boxes(T.box_str(o["box"], W, H))[0], o["area"] / (W * H)))
    return gt


def score_where(ans, gt, name):
    pb = T.parse_boxes(ans)
    return max((T.iou(pb[0], g) for g, _ in gt[name]), default=0.0) if pb else 0.0


def score_list(lst, gt, max_new):
    lines = [ln.strip() for ln in lst.splitlines() if "[" in ln]
    preds = []
    for ln in lines:
        b = T.parse_boxes(ln)
        if b:
            preds.append((re.sub(r"\s*\[.*", "", ln).strip().lower(), b[0]))
    said = {n for n, _ in preds}
    cls_p = len(said & set(gt)) / max(1, len(said))
    cls_r = len(said & set(gt)) / len(gt)
    # box level: greedy one-to-one matching, same class, IoU >= 0.5
    used, tp = set(), 0
    for n, b in preds:
        best, bi = 0.5, None
        for i, (g, _) in enumerate(gt.get(n, [])):
            if (n, i) not in used and T.iou(b, g) >= best:
                best, bi = T.iou(b, g), i
        if bi is not None:
            used.add((n, bi)); tp += 1
    n_gt = sum(len(v) for v in gt.values())
    dup = 1 - len(set(lines)) / len(lines) if lines else 0.0
    return dict(cls_p=cls_p, cls_r=cls_r, box_p=tp / max(1, len(preds)), box_r=tp / n_gt, dup=dup,
                n_pred=len(preds), n_gt=n_gt)


@torch.no_grad()
def main():
    device = "cuda"
    meta = json.load(open(T.DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    ev = [by[s] for s in meta["split"]["eval"]][:200]
    ev = [m for m in ev if gt_of(m)]
    X = np.load(T.DATA / "x.npy", mmap_mode="r")
    tok, llm = T.load_llm(device)
    B = T.Batcher(tok, X, ev, random.Random(0))
    perm = list(range(len(ev)))
    random.Random(7).shuffle(perm)
    perm = [p if p != i else (i + 1) % len(ev) for i, p in enumerate(perm)]

    for tag in sys.argv[1:]:
        conn = T.Connector(llm.config.hidden_size).to(device)
        conn.load_state_dict(torch.load(T.DATA / tag / "connector.pt"))

        def gen(feat_m, q, n):
            pre, post = B.prompt_parts(q)
            x = torch.from_numpy(np.asarray(X[feat_m["off"]:feat_m["off"] + feat_m["n"]], dtype=np.float32))
            emb, att, _ = T.embed_batch(llm, conn, [(pre, x, post, [])], device)
            out = llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            ids = out[0].tolist()
            return tok.decode(ids, skip_special_tokens=True).strip(), len(ids) >= n

        res = {}
        for cond in ("image", "shuffled"):
            rows = []
            for i, m in enumerate(ev):
                fm = m if cond == "image" else ev[perm[i]]
                gt = gt_of(m)
                name = max(gt, key=lambda n: max(a for _, a in gt[n]))
                size = max(a for _, a in gt[name])
                w, _ = gen(fm, T.Q_WHERE.format(name), 60)
                lst, trunc = gen(fm, T.Q_LIST, 160)
                rows.append(dict(stem=m["stem"], name=name, size=size, where=w, where_iou=score_where(w, gt, name),
                                 list=lst, trunc=trunc, **score_list(lst, gt, 160)))
                if (i + 1) % 50 == 0:
                    print(tag, cond, i + 1, f"iou {np.mean([r['where_iou'] for r in rows]):.3f}", flush=True)
            ious = np.array([r["where_iou"] for r in rows])
            summ = dict(n=len(rows), where_iou=float(ious.mean()), where_acc50=float((ious >= .5).mean()))
            for s, lo, hi in SIZES:
                sel = [r["where_iou"] for r in rows if lo <= r["size"] < hi]
                summ[f"where_acc50_{s}"] = float(np.mean(np.array(sel) >= .5)) if sel else None
                summ[f"n_{s}"] = len(sel)
            for k in ("cls_p", "cls_r", "box_p", "box_r", "dup", "trunc", "n_pred", "n_gt"):
                summ[k] = float(np.mean([r[k] for r in rows]))
            res[cond] = dict(summary=summ, rows=rows)
            print(tag, cond, json.dumps(summ), flush=True)
        json.dump(res, open(T.DATA / tag / "eval_extra.json", "w"), indent=1)


if __name__ == "__main__":
    main()
