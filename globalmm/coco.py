"""
COCO train2017 data for training a projector: annotations, images, and the training tasks.

Annotations are downloaded once (about 250 MB) and compacted into one small .npz file. Images are
fetched one by one and shrunk on arrival (longest side 448, sides multiples of 32), which needs
about 4 GB instead of the 18 GB of the original images. A folder of original train2017 JPEGs can
be used instead with --coco-images; those are resized while training.

Each training example is one image with one of three tasks:
  45%  "List the objects in this image with their bounding boxes."  ->  "person [437,83,785,900]\n..."
  35%  "Where is the {category}? Answer with bounding boxes."        ->  "[322,123,700,463] ..."
  20%  "Describe this image in one sentence."                        ->  a COCO caption
Boxes are [x1,y1,x2,y2] on a 0-999 scale relative to the image.
"""

import io
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/train2017/{}.jpg"

Q_LIST = "List the objects in this image with their bounding boxes."
Q_DESC = "Describe this image in one sentence."
Q_WHERE = "Where is the {}? Answer with bounding boxes."


def fit(img: Image.Image, side: int = 448, unit: int = 32) -> Image.Image:
    s = side / max(img.size)
    return img.resize(tuple(max(unit, round(v * s / unit) * unit) for v in img.size), Image.BICUBIC)


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def ensure_annotations(root: Path) -> Path:
    """Download and compact COCO train2017 annotations. Returns the path of train_meta.npz."""
    meta = root / "train_meta.npz"
    if meta.exists():
        return meta
    ann = root / "annotations"
    inst, caps = ann / "instances_train2017.json", ann / "captions_train2017.json"
    if not (inst.exists() and caps.exists()):
        ann.mkdir(parents=True, exist_ok=True)
        z = root / "annotations_trainval2017.zip"
        if not z.exists():
            print("downloading COCO annotations (about 250 MB)...")
            _download(ANN_URL, z)
        with zipfile.ZipFile(z) as zf:
            for name in ("annotations/instances_train2017.json", "annotations/captions_train2017.json"):
                zf.extract(name, root)
    print("compacting annotations...")
    import ijson  # streaming parser: the instances file is 470 MB

    cats, images = {}, {}
    with open(inst, "rb") as f:
        for c in ijson.items(f, "categories.item"):
            cats[c["id"]] = c["name"]
    cid = {k: i for i, k in enumerate(sorted(cats))}
    with open(inst, "rb") as f:
        for im in ijson.items(f, "images.item"):
            images[im["id"]] = (im["file_name"], im["width"], im["height"])
    ids = sorted(images)
    row = {i: n for n, i in enumerate(ids)}
    per = [[] for _ in ids]
    with open(inst, "rb") as f:
        for a in ijson.items(f, "annotations.item", use_float=True):
            if not a["iscrowd"]:
                x, y, w, h = a["bbox"]
                per[row[a["image_id"]]].append((cid[a["category_id"]], x, y, w, h, a["area"]))
    capl = [[] for _ in ids]
    with open(caps, "rb") as f:
        for a in ijson.items(f, "annotations.item"):
            capl[row[a["image_id"]]].append(a["caption"].strip())
    np.savez(meta, stems=np.array([images[i][0][:-4] for i in ids]),
             W=np.array([images[i][1] for i in ids]), H=np.array([images[i][2] for i in ids]),
             obj_off=np.cumsum([0] + [len(p) for p in per]),
             objs=np.array([o for p in per for o in p], dtype=np.float32).reshape(-1, 6),
             cap_off=np.cumsum([0] + [len(c) for c in capl]),
             caps=np.frombuffer("\n".join(c.replace("\n", " ") for cs in capl for c in cs).encode(), np.uint8),
             cats=np.array([cats[k] for k in sorted(cats)]))
    return meta


def ensure_images(stems: list[str], out: Path, threads: int = 16) -> None:
    """Fetch missing train2017 images, shrunk to the training size on arrival. Resumable."""
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out.glob("*.jpg")}
    todo = [s for s in stems if s not in have]
    if not todo:
        return
    print(f"fetching {len(todo)} COCO images into {out} (resized, about 40 KB each)...")

    def get(stem):
        for attempt in range(5):
            try:
                with urllib.request.urlopen(IMG_URL.format(stem), timeout=30) as r:
                    img = Image.open(io.BytesIO(r.read())).convert("RGB")
                tmp = out / f".{stem}.tmp"
                fit(img).save(tmp, "JPEG", quality=95)
                tmp.rename(out / f"{stem}.jpg")
                return True
            except Exception:
                time.sleep(2 * (attempt + 1))
        return False

    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(threads) as ex:
        for ok in ex.map(get, todo):
            done += 1
            failed += not ok
            if done % 2000 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo) - done) / rate / 60:.0f} min  failed {failed}",
                      flush=True)


class Coco:
    """Compact train2017 annotations plus the image folder."""

    def __init__(self, meta: Path, images: Path):
        z = np.load(meta)
        self.stems = z["stems"].tolist()
        self.W, self.H = z["W"], z["H"]
        self.obj_off, self.objs, self.cap_off = z["obj_off"], z["objs"], z["cap_off"]
        if "caps" in z:   # captions are stored as one newline-separated UTF-8 blob
            self.caps, self.names = z["caps"].tobytes().decode().split("\n"), z["cats"].tolist()
        else:  # older layout: captions and category names in json files next to the npz
            self.caps = json.load(open(meta.parent / "train_caps.json"))
            self.names = json.load(open(meta.parent / "cats.json"))
        self.images = images

    def available(self) -> list[int]:
        present = {p.stem for p in self.images.glob("*.jpg")}
        return [i for i, s in enumerate(self.stems) if s in present]

    def image(self, i: int) -> Image.Image:
        return Image.open(self.images / f"{self.stems[i]}.jpg").convert("RGB")

    def meta(self, i: int) -> dict:
        o = self.objs[self.obj_off[i]:self.obj_off[i + 1]]
        return dict(W=int(self.W[i]), H=int(self.H[i]),
                    objs=[dict(name=self.names[int(r[0])], box=r[1:5].tolist(), area=float(r[5])) for r in o],
                    caps=self.caps[self.cap_off[i]:self.cap_off[i + 1]])


def box_str(b, W, H) -> str:
    x, y, w, h = b
    q = lambda v, d: max(0, min(999, int(round(v / d * 999))))
    return f"[{q(x, W)},{q(y, H)},{q(x + w, W)},{q(y + h, H)}]"


def make_example(m: dict, rng) -> tuple[str, str]:
    """(question, answer) for one image, task chosen at random."""
    W, H = m["W"], m["H"]
    objs = sorted((o for o in m["objs"] if o["area"] >= 0.004 * W * H), key=lambda o: -o["area"])
    r = rng.random()
    if objs and r < 0.45:
        return Q_LIST, "\n".join(f"{o['name']} {box_str(o['box'], W, H)}" for o in objs[:12])
    if objs and r < 0.8:
        name = rng.choice(sorted({o["name"] for o in objs}))
        return Q_WHERE.format(name), " ".join([box_str(o["box"], W, H) for o in objs if o["name"] == name][:6])
    return Q_DESC, rng.choice(m["caps"]) if m["caps"] else "An image."
