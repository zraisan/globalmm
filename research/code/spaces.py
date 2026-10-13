"""Loaders for the three latent spaces we compare.

  A  Qwen3.8 image tokens   (mmproj / merger output, 2560-d)   analysis/qwen_full/*.npz
  B  Qwen3.8 word embeddings (input embedding table, 2560-d)   dl/qwen38fn_embed.raw (BF16)
  C  Llama-3.2-1B word embeddings (2048-d)                       dl/llama32-1b_embed.raw (BF16)

Word tables are memory-mapped and only the requested rows are converted to float32.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # .globalmm
DL = ROOT / "dl"
AN = ROOT / "analysis"


def _bf16_table(stem: str) -> np.memmap:
    meta = json.load(open(DL / f"{stem}.json"))
    assert meta["dtype"] == "BF16"
    return np.memmap(DL / f"{stem}.raw", dtype=np.uint16, mode="r", shape=tuple(meta["shape"]))


def _bf16_rows(table: np.memmap, ids) -> np.ndarray:
    raw = np.asarray(table[np.asarray(ids)], dtype=np.uint32) << 16
    return raw.view(np.float32)


def _vocab(tokenizer_json: Path) -> dict[str, int]:
    return json.load(open(tokenizer_json))["model"]["vocab"]


class WordSpace:
    def __init__(self, name: str, stem: str, tokenizer_json: Path):
        self.name = name
        self.table = _bf16_table(stem)
        self.vocab = _vocab(tokenizer_json)
        self.dim = self.table.shape[1]

    def rows(self, ids) -> np.ndarray:
        return _bf16_rows(self.table, ids)

    def word_ids(self, word: str) -> list[int] | None:
        """Ids for ' word' as one token if possible, else greedy longest-match pieces."""
        w = "Ġ" + word.replace(" ", "Ġ")
        if w in self.vocab:
            return [self.vocab[w]]
        ids, i = [], 0
        while i < len(w):
            for j in range(len(w), i, -1):
                if w[i:j] in self.vocab:
                    ids.append(self.vocab[w[i:j]])
                    i = j
                    break
            else:
                return None
        return ids

    def word_vec(self, word: str) -> np.ndarray:
        return self.rows(self.word_ids(word)).mean(0)


def qwen_words() -> WordSpace:
    return WordSpace("Qwen3.8 words", "qwen38fn_embed", DL / "qwen38fn-tokenizer.json")


def llama_words() -> WordSpace:
    return WordSpace("Llama-3.2 words", "llama32-1b_embed", DL / "llama32-1b-tokenizer.json")


def shared_english(a: WordSpace, b: WordSpace, min_len: int = 3) -> list[str]:
    """Lower-case alphabetic words that are a single ' word' token in both vocabularies."""
    pat = re.compile(rf"^Ġ[a-z]{{{min_len},}}$")
    common = [t for t in a.vocab if pat.match(t) and t in b.vocab]
    common.sort(key=lambda t: a.vocab[t])  # lower BPE id ~ more frequent
    return [t[1:] for t in common]


def coco_categories() -> tuple[list[str], list[str], dict[int, int]]:
    """names, supercategories, coco category id -> index (0..79)."""
    j = json.load(open(ROOT / "coco/annotations/instances_val2017.json"))
    cats = sorted(j["categories"], key=lambda c: c["id"])
    return [c["name"] for c in cats], [c["supercategory"] for c in cats], {c["id"]: i for i, c in enumerate(cats)}


def image_tokens():
    """All encoded images: list of dicts(stem, gh, gw, W, H, toks fp16 [gh*gw, 2560])."""
    out = []
    for f in sorted((AN / "qwen_full").glob("*.npz")):
        z = np.load(f)
        toks = z["toks"]  # each z[...] access re-reads the array, so read once
        off = 0
        for s, gh, gw, W, H in zip(z["stems"], z["gh"], z["gw"], z["W"], z["H"]):
            n = int(gh * gw)
            out.append(dict(stem=str(s), gh=int(gh), gw=int(gw), W=int(W), H=int(H), toks=toks[off:off + n]))
            off += n
    return out


def token_labels(imgs, bg_thresh: float = 0.5, obj_thresh: float = 0.5) -> list[np.ndarray]:
    """Per token: COCO category index (0..79), -1 = background, -2 = mixed/ambiguous.

    Rasterises instance polygons (largest first, so small objects win overlaps) and takes
    the majority label inside each merged-token cell.
    """
    from PIL import Image, ImageDraw

    _, _, cidx = coco_categories()
    j = json.load(open(ROOT / "coco/annotations/instances_val2017.json"))
    by_img: dict[int, list] = {}
    for a in j["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    labels = []
    for im in imgs:
        W, H, gh, gw = im["W"], im["H"], im["gh"], im["gw"]
        mask = Image.new("I", (W, H), 0)  # 0 = background, k+1 = category k
        draw = ImageDraw.Draw(mask)
        for a in sorted(by_img.get(int(im["stem"]), []), key=lambda a: -a["area"]):
            v = cidx[a["category_id"]] + 1
            if isinstance(a["segmentation"], list):
                for poly in a["segmentation"]:
                    if len(poly) >= 6:
                        draw.polygon(poly, fill=v)
            else:  # crowd RLE: fall back to its box
                x, y, w, h = a["bbox"]
                draw.rectangle([x, y, x + w, y + h], fill=v)
        m = np.asarray(mask)
        lab = np.full(gh * gw, -2, dtype=np.int16)
        ys = np.linspace(0, H, gh + 1).astype(int)
        xs = np.linspace(0, W, gw + 1).astype(int)
        for r in range(gh):
            for c in range(gw):
                cell = m[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].ravel()
                if cell.size == 0:
                    continue
                cnt = np.bincount(cell, minlength=81)
                k = cnt.argmax()
                frac = cnt[k] / cell.size
                if k == 0 and frac >= bg_thresh:
                    lab[r * gw + c] = -1
                elif k > 0 and frac >= obj_thresh:
                    lab[r * gw + c] = k - 1
        labels.append(lab)
    return labels
