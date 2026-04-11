"""
globalmm CLI.

Usage:
    python -m globalmm build --llm Qwen/Qwen2.5-1.5B-Instruct --concepts my.txt --out qwen.gguf
"""

import argparse
from pathlib import Path

from .projector import compute_W
from .build_mmproj import build_mmproj


def cmd_build(args: argparse.Namespace) -> None:
    print(f"[1/2] Computing projector W for {args.llm}")
    W = compute_W(
        llm_name=args.llm,
        patches_cache=args.patches_cache,
        pooled_cache=args.pooled_cache,
        concepts_path=args.concepts,
        images_dir=args.images,
        max_images=args.max_images,
        k=args.topk,
    )

    out_path = Path(args.out)
    if args.save_w:
        w_path = out_path.with_suffix(".W.pt")
        import torch
        torch.save(W, w_path)
        print(f"  saved W to {w_path}")

    print(f"[2/2] Writing mmproj to {out_path}")
    build_mmproj(W, out_path)


def main() -> None:
    p = argparse.ArgumentParser(prog="globalmm")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build an mmproj.gguf for a target LLM")
    b.add_argument("--llm", required=True, help="HF model name of the target LLM")
    b.add_argument("--out", required=True, help="output path for mmproj.gguf")
    b.add_argument("--concepts", required=True,
                   help="one word per line, describing the concepts the LLM should recognize")
    b.add_argument("--images", default=".globalmm/images",
                   help="image folder to encode; COCO val2017 is downloaded here on first run if empty")
    b.add_argument("--max-images", type=int, default=5000,
                   help="cap on images to encode when building the cache")
    b.add_argument("--topk", type=int, default=3, help="top-k concepts blended per image when building Y")
    b.add_argument("--patches-cache", default=".globalmm/patches.pt",
                   help="SigLIP post-pool/post-rms patch cache")
    b.add_argument("--pooled-cache", default=".globalmm/pooled.pt",
                   help="SigLIP attention-pooled image cache")
    b.add_argument("--save-w", action="store_true", help="also write the raw W tensor next to the gguf")
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
