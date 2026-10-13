"""Out-of-distribution grounding set: LVIS v1 objects whose category is NOT one of COCO's 80, on our
500 held-out val2017 eval images (never trained on by either connector). Streamed with ijson (200 MB json).

Per image: the largest non-COCO LVIS object with area >= 1% of the image; query name = LVIS synonym 0.
Writes analysis/grounding/lvis_ood.json: [{stem, name, synset, boxes [[x1,y1,x2,y2] 0..999], size}]
"""
import json
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parents[1]
meta = json.load(open(ROOT / "analysis/grounding/meta.json"))
by = {m["stem"]: m for m in meta["metas"]}
ev = {int(s): by[s] for s in meta["split"]["eval"]}
coco = {n.replace(" ", "_") for n in json.load(open(ROOT / "analysis/grounding_full/cats.json"))}
# LVIS names for COCO classes that differ from the COCO name (excluded too)
coco |= {"television_set", "sofa", "dining_table", "cellular_telephone", "hair_dryer", "ball", "flowerpot",
         "motorcycle", "airplane", "wineglass", "doughnut", "hotdog", "computer_keyboard", "mouse_(computer_equipment)",
         "remote_control", "refrigerator", "teddy_bear", "baseball_glove", "kite", "frisbee", "skateboard",
         "surfboard", "tennis_racket", "baseball_bat", "skis", "snowboard", "parking_meter", "fire_hydrant",
         "stop_sign", "traffic_light", "suitcase", "handbag", "umbrella", "backpack", "necktie", "bench",
         "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "person",
         "bicycle", "car", "bus", "train", "truck", "boat", "bottle", "cup", "fork", "knife", "spoon", "bowl",
         "banana", "apple", "sandwich", "orange_(fruit)", "broccoli", "carrot", "pizza", "cake", "chair",
         "bed", "toilet", "laptop", "microwave_oven", "oven", "toaster", "sink", "book", "clock", "vase",
         "scissors", "toothbrush", "pickup_truck", "flower_arrangement", "potted_plant"}
# final query names that are still COCO classes (or sub-types of one) under another LVIS name
NEAR = {"train", "bus", "school bus", "laptop computer", "car", "motor scooter", "kitchen sink", "monitor",
        "armchair", "rocking chair", "highchair", "jeep", "tow truck", "grizzly", "polar bear", "bullet train",
        "railcar", "gull", "lamb", "soccer ball", "ski", "water bottle", "glass", "table", "coffee table"}
f = ROOT / "coco/lvis/lvis_v1_val.json"
cats = {c["id"]: c for c in ijson.items(open(f, "rb"), "categories.item")}
best = {}
for a in ijson.items(open(f, "rb"), "annotations.item", use_float=True):
    m = ev.get(a["image_id"])
    c = cats[a["category_id"]]
    if m is None or c["name"] in coco or any(s in coco for s in c["synonyms"]) or \
            c["synonyms"][0].replace("_", " ").split(" (")[0] in NEAR:
        continue
    frac = a["area"] / (m["W"] * m["H"])
    x, y, w, h = a["bbox"]
    q = lambda v, d: max(0, min(999, int(round(v / d * 999))))
    box = [q(x, m["W"]), q(y, m["H"]), q(x + w, m["W"]), q(y + h, m["H"])]
    best.setdefault(m["stem"], {}).setdefault(c["id"], dict(boxes=[], size=0.0))
    e = best[m["stem"]][c["id"]]
    e["boxes"].append(box); e["size"] = max(e["size"], frac)
out = []
for stem, d in best.items():
    cid, e = max(d.items(), key=lambda kv: kv[1]["size"])
    if e["size"] >= 0.01:
        c = cats[cid]
        name = c["synonyms"][0].replace("_", " ")
        name = name.split(" (")[0]
        out.append(dict(stem=stem, name=name, synset=c["synset"], boxes=e["boxes"], size=e["size"]))
out.sort(key=lambda r: r["stem"])
json.dump(out, open(ROOT / "analysis/grounding/lvis_ood.json", "w"), indent=1)
print(len(out), "images;", sorted({r["name"] for r in out})[:80])
