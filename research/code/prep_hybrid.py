"""Decode cache for the hybrid connector: for every image, run Qwen3-VL-2B's vision tower + DeepStack +
its first LLM layer (frozen, copied), decode each image token with Qwen's norm + output head, and keep
the top-5 tokens mapped to Llama-3.2-1B token ids (identical BPE string) with renormalised probabilities.

The forward pass stops after layer 1 (a pre-hook on layer 1 raises), so only one LLM layer runs.
Qwen3-VL and Qwen3.8 both use 16 px patches merged 2x2 -> one token per 32x32 px, raster order, so the
grids match the connector's tokens one to one (checked: token counts must equal meta gh*gw).

usage: prep_hybrid.py eval            # the 500 val2017 eval-split images
       prep_hybrid.py train [N]       # N random train2017 images (default 32000)
-> analysis/hybrid/decode_<split>.npz: stems, off [n+1], ids [T,5] int32 (Llama ids, -1 = unmapped), p [T,5] f16
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

import train_connector as T

MD = T.ROOT / "dl/qwen3vl-2b"
OUT = T.ROOT / "analysis/hybrid"
OUT.mkdir(parents=True, exist_ok=True)
K = 1


class Stop(Exception):
    pass


def fit32(img, side=448):
    s = side / max(img.size)
    return img.resize(tuple(max(32, round(v * s / 32) * 32) for v in img.size), Image.BICUBIC)


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    split = sys.argv[1]
    if split == "eval":
        meta = json.load(open(T.DATA / "meta.json"))
        by = {m["stem"]: m for m in meta["metas"]}
        items = [(s, T.ROOT / "coco/val2017" / f"{s}.jpg", by[s]["gh"] * by[s]["gw"], True) for s in meta["split"]["eval"]]
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
        z = np.load(T.ROOT / "analysis/grounding_full/train_meta.npz")
        present = {p.stem for p in (T.ROOT / "coco/train448").glob("*.jpg")}
        stems = [s for s in z["stems"].tolist() if s in present]
        stems = [stems[i] for i in np.random.default_rng(42).permutation(len(stems))[:n]]
        items = [(s, T.ROOT / "coco/train448" / f"{s}.jpg", None, False) for s in stems]
    qv = json.load(open(MD / "tokenizer.json"))["model"]["vocab"]
    lv = json.load(open(T.ROOT / "dl/llama32-1b-tokenizer.json"))["model"]["vocab"]
    q2l = np.full(max(qv.values()) + 1 + 512, -1, np.int64)
    for s, qi in qv.items():
        if s in lv:
            q2l[qi] = lv[s]
    q2l = torch.from_numpy(q2l).cuda()

    proc = AutoProcessor.from_pretrained(MD)
    proc.image_processor.size = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}
    model = Qwen3VLForConditionalGeneration.from_pretrained(MD, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    cap = {}

    def stop_hook(m, args):
        cap["h"] = args[0].detach()
        raise Stop
    lm.layers[K].register_forward_pre_hook(stop_hook)
    img_tok = model.config.image_token_id
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}],
                                    add_generation_prompt=True)
    stems, offs, IDS, PS, bad = [], [0], [], [], 0
    with torch.no_grad():
        for j, (s, path, expect, resize) in enumerate(items):
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                bad += 1; continue
            if resize:
                img = fit32(img)
            inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            try:
                model(**inp)
            except Stop:
                pass
            pos = (inp["input_ids"][0] == img_tok).nonzero().squeeze(1)
            gh, gw = (inp["image_grid_thw"][0, 1:] // 2).tolist()
            if expect is not None and len(pos) != expect:
                bad += 1; continue
            h = cap["h"][0, pos]
            logits = model.lm_head(lm.norm(h)).float()
            p, ids = torch.softmax(logits, -1).topk(5, -1)
            lid = q2l[ids]
            p = torch.where(lid >= 0, p, torch.zeros_like(p))
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
            stems.append(s); offs.append(offs[-1] + len(pos))
            IDS.append(lid.int().cpu().numpy()); PS.append(p.half().cpu().numpy())
            if j % 2000 == 0:
                print(f"{j}/{len(items)}  grid {gh}x{gw}  bad {bad}", flush=True)
    np.savez(OUT / f"decode_{split}.npz", stems=np.array(stems), off=np.array(offs), ids=np.concatenate(IDS), p=np.concatenate(PS))
    print(f"DONE {len(stems)} images, {offs[-1]} tokens, skipped {bad}", flush=True)


if __name__ == "__main__":
    main()
