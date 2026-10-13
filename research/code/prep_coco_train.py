"""Compact COCO train2017 annotations, streamed (the instances file is 470 MB; json.load would need GBs).

Writes analysis/grounding_full/train_meta.npz:
  stems [N] str, W/H [N], obj_off [N+1], objs [M, 6] float32 = (cat_idx, x, y, w, h, area) non-crowd only,
  cap_off [N+1]; and train_caps.json (flat caption list); cats.json (80 names, cat_idx order)
"""
import json
from pathlib import Path

import ijson
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/grounding_full"
OUT.mkdir(parents=True, exist_ok=True)
ANN = ROOT / "coco/annotations"

cats, images = {}, {}
with open(ANN / "instances_train2017.json", "rb") as f:
    for c in ijson.items(f, "categories.item"):
        cats[c["id"]] = c["name"]
cid = {k: i for i, k in enumerate(sorted(cats))}
with open(ANN / "instances_train2017.json", "rb") as f:
    for im in ijson.items(f, "images.item"):
        images[im["id"]] = (im["file_name"], im["width"], im["height"])
ids = sorted(images)
row = {i: n for n, i in enumerate(ids)}
per = [[] for _ in ids]
with open(ANN / "instances_train2017.json", "rb") as f:
    for a in ijson.items(f, "annotations.item", use_float=True):
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        per[row[a["image_id"]]].append((cid[a["category_id"]], x, y, w, h, a["area"]))
caps = [[] for _ in ids]
with open(ANN / "captions_train2017.json", "rb") as f:
    for a in ijson.items(f, "annotations.item"):
        caps[row[a["image_id"]]].append(a["caption"].strip())

obj_off = np.cumsum([0] + [len(p) for p in per])
cap_off = np.cumsum([0] + [len(c) for c in caps])
objs = np.array([o for p in per for o in p], dtype=np.float32).reshape(-1, 6)
np.savez(OUT / "train_meta.npz", stems=np.array([images[i][0][:-4] for i in ids]),
         W=np.array([images[i][1] for i in ids]), H=np.array([images[i][2] for i in ids]),
         obj_off=obj_off, objs=objs, cap_off=cap_off)
json.dump([c for cs in caps for c in cs], open(OUT / "train_caps.json", "w"))
json.dump([cats[k] for k in sorted(cats)], open(OUT / "cats.json", "w"))
print(f"{len(ids)} images, {len(objs)} objects, {cap_off[-1]} captions, {sum(1 for p in per if not p)} without objects")
