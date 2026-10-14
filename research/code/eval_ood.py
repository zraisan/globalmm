"""Out-of-distribution grounding: "Where is the X?" for non-COCO LVIS objects (lvis_ood.json, from
prep_lvis.py) on held-out val2017 images. Conditions: image / shuffled (another image's features).

usage: eval_ood.py TAG [TAG...]   -> analysis/grounding/<TAG>/eval_ood.json
"""
import json
import random
import sys

import numpy as np
import torch

import eval_extra as E
import train_connector as T


def summarize(rows):
    ious = np.array([r["iou"] for r in rows])
    s = dict(n=len(rows), where_iou=float(ious.mean()), where_acc50=float((ious >= .5).mean()))
    for k, lo, hi in [("medium", 0, 0.10), ("large", 0.10, 1.01)]:
        sel = np.array([r["iou"] for r in rows if lo <= r["size"] < hi])
        s[f"where_acc50_{k}"] = float((sel >= .5).mean()) if len(sel) else None
    s["no_box"] = float(np.mean([r["no_box"] for r in rows]))
    return s


@torch.no_grad()
def main():
    ood = json.load(open(T.DATA / "lvis_ood.json"))
    by = {m["stem"]: m for m in json.load(open(T.DATA / "meta.json"))["metas"]}
    X = np.load(T.DATA / "x.npy", mmap_mode="r")
    tok, llm = T.load_llm("cuda")
    B = T.Batcher(tok, X, [], random.Random(0))
    perm = [(i + 7) % len(ood) for i in range(len(ood))]
    for tag in sys.argv[1:]:
        conn = T.Connector(llm.config.hidden_size).cuda()
        conn.load_state_dict(torch.load(T.DATA / tag / "connector.pt"))
        res = {}
        for cond in ("image", "shuffled"):
            rows = []
            for i, r in enumerate(ood):
                fm = by[r["stem"] if cond == "image" else ood[perm[i]]["stem"]]
                pre, post = B.prompt_parts(T.Q_WHERE.format(r["name"]))
                x = torch.from_numpy(np.asarray(X[fm["off"]:fm["off"] + fm["n"]], dtype=np.float32))
                emb, att, _ = T.embed_batch(llm, conn, [(pre, x, post, [])], "cuda")
                o = llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
                ans = tok.decode(o[0], skip_special_tokens=True).strip()
                pb = T.parse_boxes(ans)
                iou = max(T.iou(pb[0], tuple(g)) for g in r["boxes"]) if pb else 0.0
                rows.append(dict(stem=r["stem"], name=r["name"], size=r["size"], answer=ans, iou=iou, no_box=not pb))
            res[cond] = dict(summary=summarize(rows), rows=rows)
            print(tag, cond, json.dumps(res[cond]["summary"]), flush=True)
        json.dump(res, open(T.DATA / tag / "eval_ood.json", "w"), indent=1)


if __name__ == "__main__":
    main()
