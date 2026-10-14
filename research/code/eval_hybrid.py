"""Evaluate hybrid vs control connectors on the same held-out tests as before:
COCO "where is X" + object lists (192 images, eval_extra.py scoring) and the non-COCO LVIS set
(246 questions, eval_ood.py scoring).

usage: eval_hybrid.py TAG:words|nowords [...]   e.g. eval_hybrid.py hybrid:words control:nowords
-> analysis/grounding/<TAG>/eval_hybrid.json
"""
import json
import sys

import numpy as np
import torch

import eval_extra as E
import eval_ood as O
import train_connector as T
import train_hybrid as H


@torch.no_grad()
def main():
    meta = json.load(open(T.DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    coco = [m for m in [by[s] for s in meta["split"]["eval"]][:200] if E.gt_of(m)]
    lvis = json.load(open(T.DATA / "lvis_ood.json"))
    X = np.load(T.DATA / "x.npy", mmap_mode="r")
    dec = H.Decode("eval")
    tok, llm = T.load_llm("cuda")
    Emb = llm.get_input_embeddings().weight
    B = T.Batcher(tok, X, [], None)
    for arg in sys.argv[1:]:
        tag, mode = arg.split(":")
        words = mode == "words"
        conn = T.Connector(2048).cuda()
        conn.load_state_dict(torch.load(T.DATA / tag / "connector.pt"))

        def gen(m, q, n):
            pre, post = B.prompt_parts(q)
            x = torch.from_numpy(np.asarray(X[m["off"]:m["off"] + m["n"]], dtype=np.float32)).cuda()
            emb, att, _ = H.embed_hybrid(llm, conn, [(pre, x, dec.vectors(m["stem"], Emb), post, [])], words=words)
            o = llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(o[0], skip_special_tokens=True).strip()

        rows = []
        for m in coco:
            gt = E.gt_of(m)
            name = max(gt, key=lambda n: max(a for _, a in gt[n]))
            w = gen(m, T.Q_WHERE.format(name), 60)
            lst = gen(m, T.Q_LIST, 160)
            rows.append(dict(stem=m["stem"], name=name, size=max(a for _, a in gt[name]), where=w,
                             where_iou=E.score_where(w, gt, name), list=lst, **E.score_list(lst, gt, 160)))
        ious = np.array([r["where_iou"] for r in rows])
        cs = dict(n=len(rows), where_iou=float(ious.mean()), where_acc50=float((ious >= .5).mean()),
                  **{k: float(np.mean([r[k] for r in rows])) for k in ("cls_p", "cls_r", "box_p", "box_r", "dup")})
        orows = []
        for r in lvis:
            ans = gen(by[r["stem"]], T.Q_WHERE.format(r["name"]), 60)
            pb = T.parse_boxes(ans)
            orows.append(dict(stem=r["stem"], name=r["name"], size=r["size"], answer=ans, no_box=not pb,
                              iou=max(T.iou(pb[0], tuple(g)) for g in r["boxes"]) if pb else 0.0))
        osum = O.summarize(orows)
        print(tag, mode, "COCO", json.dumps(cs), "| LVIS", json.dumps(osum), flush=True)
        json.dump(dict(coco=cs, lvis=osum, coco_rows=rows, lvis_rows=orows), open(T.DATA / tag / "eval_hybrid.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
