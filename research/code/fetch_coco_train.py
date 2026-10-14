"""Download COCO train2017 images one by one, shrinking each to the training size on arrival.

Longest side 448, both sides multiples of 32 (same as fit32 everywhere else), JPEG q95.
~40 KB per image instead of ~160 KB; resumable (skips files already present).
usage: fetch_coco_train.py OUT_DIR [threads]
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
stems = np.load(ROOT / "analysis/grounding_full/train_meta.npz")["stems"].tolist()
have = {p.stem for p in out.glob("*.jpg")}
todo = [s for s in stems if s not in have]
print(f"{len(stems)} images, {len(have)} present, {len(todo)} to fetch", flush=True)
sess = requests.Session()
sess.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=threads))


def fit32(img, side=448):
    s = side / max(img.size)
    return img.resize(tuple(max(32, round(v * s / 32) * 32) for v in img.size), Image.BICUBIC)


def get(stem):
    for attempt in range(5):
        try:
            r = sess.get(f"http://images.cocodataset.org/train2017/{stem}.jpg", timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            tmp = out / f".{stem}.tmp"
            fit32(img).save(tmp, "JPEG", quality=95)
            tmp.rename(out / f"{stem}.jpg")
            return True
        except Exception as e:
            err = e
            time.sleep(2 * (attempt + 1))
    print("FAIL", stem, err, flush=True)
    return False


t0, done = time.time(), 0
with ThreadPoolExecutor(threads) as ex:
    for ok in ex.map(get, todo):
        done += 1
        if done % 2000 == 0:
            el = time.time() - t0
            print(f"{done}/{len(todo)}  {done / el:.1f} img/s  eta {(len(todo) - done) / (done / el) / 60:.0f} min", flush=True)
print("DONE", flush=True)
