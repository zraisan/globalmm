import argparse
from pathlib import Path

import torch

from globalmm.projector import compute_W
from globalmm.build_mmproj import build_mmproj


def main() -> None:
    p = argparse.ArgumentParser(prog="globalmm", description="Build llama.cpp mmproj.gguf for any LLM via lstsq.")
    p.add_argument("--llm", required=True, help="Hugging Face model name of the target LLM")
    p.add_argument("--concepts", required=True, help="one concept word per line")
    p.add_argument("--out", required=True, help="output path for the mmproj.gguf file")
    p.add_argument("--images", default=".globalmm/images",
                   help="image folder to encode; COCO val2017 is downloaded here on first run if empty")
    p.add_argument("--max-images", type=int, default=5000)
    p.add_argument("--topk", type=int, default=3, help="concepts blended per image when building Y")
    p.add_argument("--patches-cache", default=".globalmm/patches.pt")
    p.add_argument("--pooled-cache", default=".globalmm/pooled.pt")
    p.add_argument("--save-w", action="store_true", help="also write the raw W tensor next to the gguf")
    args = p.parse_args()

    W = compute_W(
        llm_name=args.llm,
        concepts_path=args.concepts,
        patches_cache=args.patches_cache,
        pooled_cache=args.pooled_cache,
        images_dir=args.images,
        max_images=args.max_images,
        k=args.topk,
    )

    out = Path(args.out)
    if args.save_w:
        torch.save(W, out.with_suffix(".W.pt"))

    build_mmproj(W, out)
