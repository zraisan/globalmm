import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] != ["build"]:   # `globalmm --llm ...` keeps working as `globalmm build --llm ...`
        argv = ["build"] + argv
    p = argparse.ArgumentParser(prog="globalmm", description="Build llama.cpp mmproj.gguf files for any LLM.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build an mmproj.gguf")
    b.add_argument("--method", choices=["lstsq", "connector"], default="lstsq",
                   help="lstsq: closed-form matrix on SigLIP, seconds on CPU (default). "
                        "connector: train a new projector for an existing mmproj, hours on a GPU, gives bounding boxes")
    b.add_argument("--llm", required=True, help="Hugging Face model name of the target LLM")
    b.add_argument("--out", required=True, help="output path for the mmproj.gguf file")

    g = b.add_argument_group("lstsq")
    g.add_argument("--concepts", help="one concept word per line (required for lstsq)")
    g.add_argument("--images", default=".globalmm/images",
                   help="image folder to encode; COCO val2017 is downloaded here on first run if empty")
    g.add_argument("--max-images", type=int, default=5000)
    g.add_argument("--topk", type=int, default=3, help="concepts blended per image when building Y")
    g.add_argument("--patches-cache", default=".globalmm/patches.pt")
    g.add_argument("--pooled-cache", default=".globalmm/pooled.pt")
    g.add_argument("--save-w", action="store_true", help="also write the raw W tensor next to the gguf")

    c = b.add_argument_group("connector")
    c.add_argument("--mmproj", help="existing mmproj.gguf whose vision tower is kept (required for connector)")
    c.add_argument("--coco-dir", default=".globalmm/coco", help="COCO annotations and resized images are kept here")
    c.add_argument("--coco-images", help="use an existing folder of COCO train2017 JPEGs instead of downloading")
    c.add_argument("--train-images", type=int, help="train on a random subset of this many images")
    c.add_argument("--steps", type=int, help="training steps (default: one pass over the images)")
    c.add_argument("--batch", type=int, default=2, help="images per forward pass")
    c.add_argument("--accum", type=int, default=4, help="forward passes per optimizer step")
    c.add_argument("--lr", type=float, default=3e-4)
    c.add_argument("--work-dir", help="checkpoints, resumable (default: .globalmm/runs/<out name>)")
    args = p.parse_args(argv)

    out = Path(args.out)
    if args.method == "lstsq":
        if not args.concepts:
            p.error("--method lstsq needs --concepts")
        from globalmm.build_mmproj import build_mmproj
        from globalmm.projector import compute_W

        W = compute_W(
            llm_name=args.llm,
            concepts_path=args.concepts,
            patches_cache=args.patches_cache,
            pooled_cache=args.pooled_cache,
            images_dir=args.images,
            max_images=args.max_images,
            k=args.topk,
        )
        if args.save_w:
            torch.save(W, out.with_suffix(".W.pt"))
        build_mmproj(W, out)
    else:
        if not args.mmproj:
            p.error("--method connector needs --mmproj")
        from globalmm.connector import train_connector

        train_connector(
            mmproj=Path(args.mmproj),
            llm_name=args.llm,
            out=out,
            coco_dir=Path(args.coco_dir),
            coco_images=Path(args.coco_images) if args.coco_images else None,
            work_dir=Path(args.work_dir) if args.work_dir else Path(".globalmm/runs") / out.stem,
            steps=args.steps,
            batch=args.batch,
            accum=args.accum,
            lr=args.lr,
            max_images=args.train_images,
        )
