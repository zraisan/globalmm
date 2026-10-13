"""Cache the input of Qwen3.8's merger (mm.0) for every image, on the GPU.

That input = vision tower -> post-layernorm -> 2x2 patches concatenated (4608-d), i.e. exactly
what llama.cpp's qwen3vl_merger feeds into mm.0. Training the connector then never needs the
vision tower again.

Images are first resized so the longest side is 448 and both sides are multiples of 32, so the
processor does no further resizing; the same resize is applied at test time before llama.cpp.

Writes OUT/shard_XXXX.npz: stems, gh, gw, W, H (original size), offs, x (fp16 [sum tokens, 4608])
usage: cache_merger_inputs.py DL_DIR IMAGE_DIR OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionConfig, Qwen3_5VisionModel

D, img_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
SIDE = 448


def fit32(img: Image.Image) -> Image.Image:
    s = SIDE / max(img.size)
    w, h = (max(32, round(v * s / 32) * 32) for v in img.size)
    return img.resize((w, h), Image.BICUBIC)


cfg = json.load(open(D / "qwen38fn_cfg/config.json"))["vision_config"]
cfg.pop("model_type", None)
with torch.device("cuda"):  # build on the GPU and stream weights in: keeps host RAM low
    model = Qwen3_5VisionModel(Qwen3_5VisionConfig(**cfg))
own = model.state_dict()
with torch.no_grad(), safe_open(D / "qwen38fn-00001.safetensors", framework="pt", device="cuda") as f:
    for k in f.keys():
        if k.startswith("model.visual.") and k[13:] in own:
            own[k[13:]].copy_(f.get_tensor(k))
del own
model = model.eval()
proc = AutoImageProcessor.from_pretrained(D / "qwen38fn_cfg")
proc.size = {"longest_edge": 4096 * 4096, "shortest_edge": 32 * 32}

grab = {}
model.merger.linear_fc1.register_forward_pre_hook(lambda m, a: grab.__setitem__("x", a[0].detach()))

done = set()
for f in out.glob("shard_*.npz"):
    done.update(np.load(f)["stems"].tolist())
paths = [p for p in sorted(img_dir.glob("*.jpg")) if p.stem not in done]
print(f"{len(done)} cached, {len(paths)} to go", flush=True)
shard = 1 + max([int(f.stem.split("_")[1]) for f in out.glob("shard_*.npz")], default=-1)
for f in out.glob("tmp_*.npz"):
    f.unlink()
buf, t0 = [], time.time()


def flush():
    global buf, shard
    if not buf:
        return
    offs = np.cumsum([0] + [len(b[5]) for b in buf])
    tmp = out / f"tmp_{shard:04d}.npz"
    np.savez(tmp, stems=np.array([b[0] for b in buf]), gh=np.array([b[1] for b in buf]),
             gw=np.array([b[2] for b in buf]), W=np.array([b[3] for b in buf]), H=np.array([b[4] for b in buf]),
             offs=offs, x=np.concatenate([b[5] for b in buf]))
    tmp.rename(out / f"shard_{shard:04d}.npz")  # atomic: a killed job never leaves a half-written shard
    shard += 1
    buf = []


for i, p in enumerate(paths):
    try:
        img = Image.open(p).convert("RGB")
    except OSError as e:
        print("skip", p.name, e, flush=True)
        continue
    small = fit32(img)
    inp = proc(images=[small], return_tensors="pt")
    _, h, w = inp["image_grid_thw"][0].tolist()
    assert (w * 16, h * 16) == small.size, (small.size, h, w)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(inp["pixel_values"].to("cuda"), grid_thw=inp["image_grid_thw"].to("cuda"))
    x = grab["x"].float().cpu().numpy().astype(np.float16)
    assert len(x) == (h // 2) * (w // 2)
    buf.append((p.stem, h // 2, w // 2, img.width, img.height, x))
    if len(buf) == 50:
        flush()
        el = time.time() - t0
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(paths)}  {el / (i + 1):.3f}s/img", flush=True)
flush()
print("DONE", flush=True)
