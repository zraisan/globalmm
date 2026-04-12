"""
First-run bootstrap: download an image dataset and encode it through SigLIP to
build the two cache files that `compute_W` needs.

Default dataset is COCO val2017 (~1 GB from cocodataset.org). Users can point at
their own image folder via `--images` on the CLI.
"""

from __future__ import annotations

import gc
import zipfile
from pathlib import Path
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SiglipModel, SiglipProcessor


SIGLIP_NAME = "google/siglip-so400m-patch14-384"
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def ensure_images(images_dir: Path, min_count: int = 1000) -> Path:
    """Make sure there are images at `images_dir`. If not, download COCO val2017.

    Returns the directory containing the images (may be a subdir of `images_dir`
    if the archive extracted with its own folder structure).
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.png"))
    if len(existing) >= min_count:
        print(f"  found {len(existing)} images at {images_dir}")
        return images_dir

    # Maybe they're in a nested subdir from a previous download
    for sub in images_dir.iterdir() if images_dir.exists() else []:
        if sub.is_dir():
            n = len(list(sub.glob("*.jpg")))
            if n >= min_count:
                print(f"  found {n} images at {sub}")
                return sub

    print(f"  no local images at {images_dir}; downloading COCO val2017 (~1 GB)...")
    zip_target = images_dir / "val2017.zip"
    _download(COCO_VAL2017_URL, zip_target)

    print(f"  extracting {zip_target.name}...")
    with zipfile.ZipFile(zip_target) as zf:
        zf.extractall(images_dir)
    zip_target.unlink()

    # COCO zip extracts into `val2017/`
    nested = images_dir / "val2017"
    if nested.exists() and any(nested.glob("*.jpg")):
        return nested
    return images_dir


def _download(url: str, target: Path, chunk: int = 1 << 20) -> None:
    """Stream a URL to disk with a simple progress line."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        with open(target, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total:
                    pct = 100 * got / total
                    print(f"\r    {got / 1e6:.0f}/{total / 1e6:.0f} MB  ({pct:.1f}%)", end="", flush=True)
    print()


def encode_dataset(
    images_dir: Path,
    patches_cache: Path,
    pooled_cache: Path,
    max_images: int = 5000,
    batch_size: int = 16,
) -> None:
    """Encode every image in `images_dir` through SigLIP and write two cache files.

    - patches_cache: dict{features: (N,81,1152) fp16, stems: list[str]} — post
      avg_pool2d(3) + rms_norm, matching the gemma3 projector's pre-W pipeline.
    - pooled_cache:  dict{embeddings: (N,1152) fp32, stems: list[str]} — SigLIP's
      attention-pooled image output, used for zero-shot concept labeling.
    """
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.png"))
    image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError(f"no images found in {images_dir}")
    print(f"  encoding {len(image_paths)} images on {DEVICE} ...")

    model = SiglipModel.from_pretrained(SIGLIP_NAME).eval().to(DEVICE)
    processor = SiglipProcessor.from_pretrained(SIGLIP_NAME)
    vision = model.vision_model
    config = model.config.vision_config
    grid = config.image_size // config.patch_size  # 27
    n_merge = 3

    pooled_out: list[torch.Tensor] = []
    patches_out: list[torch.Tensor] = []
    stems: list[str] = []

    def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"    skip {p.name}: {e}")
                    continue
            if not imgs:
                continue
            pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)

            vm_out = vision(pixel_values=pixel_values)
            pooled = vm_out.pooler_output
            pooled_out.append(pooled.float().cpu())

            patches = vm_out.last_hidden_state
            B, _, D = patches.shape
            p = patches.reshape(B, grid, grid, D).permute(0, 3, 1, 2)
            p = F.avg_pool2d(p, kernel_size=n_merge, stride=n_merge)
            p = p.permute(0, 2, 3, 1).reshape(B, -1, D)
            p = rms_norm(p)
            patches_out.append(p.half().cpu())

            stems.extend(bp.stem for bp in batch_paths[: len(imgs)])
            if (i // batch_size) % 10 == 0:
                done = i + len(imgs)
                print(f"    {done}/{len(image_paths)}")

    del model, vision, processor
    _free()

    pooled_tensor = torch.cat(pooled_out, dim=0)
    patches_tensor = torch.cat(patches_out, dim=0)
    print(f"  pooled: {tuple(pooled_tensor.shape)}  patches: {tuple(patches_tensor.shape)}")

    patches_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": patches_tensor, "stems": stems}, patches_cache)
    torch.save({"embeddings": pooled_tensor, "stems": stems}, pooled_cache)
    print(f"  wrote {patches_cache} and {pooled_cache}")


def ensure_caches(
    images_dir: Path,
    patches_cache: Path,
    pooled_cache: Path,
    max_images: int = 5000,
) -> None:
    """Make sure both caches exist; encode from scratch if either is missing."""
    if patches_cache.exists() and pooled_cache.exists():
        return
    images_dir = ensure_images(images_dir)
    encode_dataset(images_dir, patches_cache, pooled_cache, max_images=max_images)
