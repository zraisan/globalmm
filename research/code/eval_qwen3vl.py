"""Reference point: a real VLM (Qwen3-VL-2B-Instruct, trained end-to-end for grounding) on the same
held-out "where is X" questions as eval_extra.py, using its native grounding prompt (JSON, 0..1000 coords).
Image resized like ours (longest side 448, multiples of 32).

usage: eval_qwen3vl.py [coco|lvis]   -> analysis/grounding/qwen3vl_ref[_lvis].json
"""
import json
import re
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import eval_extra as E
import train_connector as T

MD = T.ROOT / "dl/qwen3vl-2b"
SIZES = E.SIZES


def fit32(img, side=448):
    s = side / max(img.size)
    return img.resize(tuple(max(32, round(v * s / 32) * 32) for v in img.size), Image.BICUBIC)


@torch.no_grad()
def main():
    meta = json.load(open(T.DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    mode = sys.argv[1] if len(sys.argv) > 1 else "coco"
    if mode == "coco":
        ev = [m for m in [by[s] for s in meta["split"]["eval"]][:200] if E.gt_of(m)]
        qs = []
        for m in ev:
            gt = E.gt_of(m)
            name = max(gt, key=lambda n: max(a for _, a in gt[n]))
            qs.append((m, name, [g for g, _ in gt[name]], max(a for _, a in gt[name])))
    else:
        qs = [(by[r["stem"]], r["name"], [tuple(b) for b in r["boxes"]], r["size"])
              for r in json.load(open(T.DATA / "lvis_ood.json"))]
    proc = AutoProcessor.from_pretrained(MD)
    proc.image_processor.size = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}
    model = Qwen3VLForConditionalGeneration.from_pretrained(MD, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = []
    for i, (m, name, gtb, size) in enumerate(qs):
        img = fit32(Image.open(T.ROOT / "coco/val2017" / f"{m['stem']}.jpg").convert("RGB"))
        q = f"Locate the {name} in the image and output its bbox coordinates in JSON format."
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        out = model.generate(**inp, max_new_tokens=80, do_sample=False)
        ans = proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        b = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", ans)
        iou = 0.0
        if b:
            p = tuple(round(int(v) * 999 / 1000) for v in b[0])
            iou = max(T.iou(p, g) for g in gtb)
        rows.append(dict(stem=m["stem"], name=name, size=size, answer=ans, where_iou=iou))
        if (i + 1) % 50 == 0:
            print(i + 1, f"iou {np.mean([r['where_iou'] for r in rows]):.3f}", flush=True)
    ious = np.array([r["where_iou"] for r in rows])
    summ = dict(n=len(rows), where_iou=float(ious.mean()), where_acc50=float((ious >= .5).mean()))
    for s, lo, hi in SIZES:
        sel = [r["where_iou"] for r in rows if lo <= r["size"] < hi]
        summ[f"where_acc50_{s}"] = float(np.mean(np.array(sel) >= .5)) if sel else None
    print(json.dumps(summ), flush=True)
    json.dump(dict(summary=summ, rows=rows), open(T.DATA / ("qwen3vl_ref.json" if mode == "coco" else "qwen3vl_ref_lvis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
