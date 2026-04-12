"""
Compute the W projector (SigLIP features -> target LLM embeddings) via lstsq.

The `lin3` recipe — faithful to the one validated at 7/8 on the test set:

  1. Encode concept words with SigLIP's text encoder -> concept_siglip (N_c, 1152)
  2. Look each concept up in the target LLM's input embedding table, mean-pooling
     subwords -> concept_llm (N_c, d_llm)
  3. Load cached patch features (N, 81, 1152) and take the per-image MEAN over the
     81 patches -> X (N, 1152). One vector per image.
  4. For each image, SigLIP zero-shot top-3 against concept_siglip, linear-normalize
     the similarities, blend concept_llm rows -> Y (N, d_llm)
  5. Word-balance: cap the number of images per primary concept at 50 to avoid the
     COCO distribution (lots of people, few elephants) skewing W.
  6. W = lstsq(X_bal, Y_bal).solution, shape (1152, d_llm).

W is a single 1152xd_llm matrix applied by llama.cpp to every patch at inference,
even though it was fit on per-image means — a linear map is shape-agnostic.
"""

import gc
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, SiglipTextModel

from .encode import ensure_caches


SIGLIP_NAME = "google/siglip-so400m-patch14-384"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _load_concepts(path: Path) -> list[str]:
    return [w.strip() for w in path.read_text().splitlines() if w.strip()]


def _encode_concepts_siglip(concepts: list[str]) -> torch.Tensor:
    """Encode concept words with SigLIP's text tower only. Frees SigLIP before returning."""
    print(f"  loading SigLIP text tower on {DEVICE}...")
    text_model = SiglipTextModel.from_pretrained(SIGLIP_NAME).eval().to(DEVICE)
    tok = AutoTokenizer.from_pretrained(SIGLIP_NAME)
    prompts = [f"a photo of a {c}" for c in concepts]
    print(f"  encoding {len(prompts)} concepts...")
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(prompts), 128):
            batch = tok(prompts[i : i + 128], return_tensors="pt", padding="max_length", truncation=True)
            ids = batch["input_ids"].to(DEVICE)
            emb = text_model(input_ids=ids).pooler_output
            out.append(emb.float().cpu())
    embs = torch.cat(out, dim=0)
    del text_model, tok, out
    _free()
    return F.normalize(embs, dim=-1)


def _load_embedding_table(llm_name: str) -> torch.Tensor:
    """Read just the input embedding tensor from the HF model's safetensors shards.

    Avoids loading the full LLM into RAM. Tries common tensor names used by
    Qwen/Llama/Mistral-family causal LMs.
    """
    candidates = [
        "model.embed_tokens.weight",   # Qwen2/Llama/Mistral
        "embed_tokens.weight",
        "transformer.wte.weight",      # GPT2/Falcon
        "tok_embeddings.weight",       # older Llama
        "gpt_neox.embed_in.weight",    # Pythia
    ]
    try:
        index_path = hf_hub_download(llm_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        for name in candidates:
            if name in weight_map:
                shard = hf_hub_download(llm_name, weight_map[name])
                with safe_open(shard, framework="pt") as sf:
                    return sf.get_tensor(name).float()
    except Exception:
        pass

    single = hf_hub_download(llm_name, "model.safetensors")
    with safe_open(single, framework="pt") as sf:
        keys = list(sf.keys())
        for name in candidates:
            if name in keys:
                return sf.get_tensor(name).float()
        for k in keys:
            if "embed" in k.lower() and "weight" in k:
                return sf.get_tensor(k).float()
    raise RuntimeError(f"Could not find input embedding tensor in {llm_name}")


def _lookup_concepts_llm(
    concepts: list[str], llm_name: str
) -> tuple[torch.Tensor, list[int], int]:
    """Extract concept embeddings from the target LLM's input embedding table."""
    print(f"  loading target LLM embedding table: {llm_name}")
    tok = AutoTokenizer.from_pretrained(llm_name)
    emb_table = _load_embedding_table(llm_name)
    d_llm = emb_table.shape[1]
    print(f"    embed table: {tuple(emb_table.shape)}")

    kept_vecs: list[torch.Tensor] = []
    kept_idx: list[int] = []
    for i, word in enumerate(concepts):
        ids = tok(" " + word, add_special_tokens=False).input_ids
        if not ids:
            continue
        vecs = emb_table[torch.tensor(ids)]
        kept_vecs.append(vecs.mean(dim=0))
        kept_idx.append(i)
    concept_llm = torch.stack(kept_vecs, dim=0)

    del emb_table, tok, kept_vecs
    _free()
    return concept_llm, kept_idx, d_llm


def compute_W(
    llm_name: str,
    concepts_path: str | Path,
    patches_cache: str | Path = ".globalmm/patches.pt",
    pooled_cache: str | Path = ".globalmm/pooled.pt",
    images_dir: str | Path = ".globalmm/images",
    max_images: int = 5000,
    k: int = 3,
    max_per_word: int = 50,
) -> torch.Tensor:
    """
    Fit the (1152, d_llm) projector W for the given LLM via the lin3 lstsq recipe.

    If the cache files are missing this will download an image dataset
    (COCO val2017 by default) and run the SigLIP encoding pass once.
    """
    ensure_caches(Path(images_dir), Path(patches_cache), Path(pooled_cache), max_images=max_images)

    concepts = _load_concepts(Path(concepts_path))
    concept_siglip_all = _encode_concepts_siglip(concepts)

    concept_llm, kept_idx, d_llm = _lookup_concepts_llm(concepts, llm_name)
    concept_siglip = concept_siglip_all[kept_idx].contiguous()
    kept_concepts = [concepts[i] for i in kept_idx]
    del concept_siglip_all
    _free()
    print(f"  concepts kept after LLM tokenizer: {len(kept_idx)}/{len(concepts)}")
    print(f"  target LLM embedding dim: {d_llm}")

    pooled = torch.load(pooled_cache, map_location="cpu")
    img_pooled = pooled["embeddings"].float()
    pooled_stems = pooled["stems"]
    del pooled
    _free()

    patches = torch.load(patches_cache, map_location="cpu")
    assert patches["stems"] == pooled_stems, "stem order mismatch between caches"
    X_mean = patches["features"].float().mean(dim=1)
    del patches
    _free()
    N = X_mean.shape[0]
    print(f"  per-image mean-patch features: {tuple(X_mean.shape)}")

    # Zero-shot labeling: similarity against concept_siglip
    img_norm = F.normalize(img_pooled, dim=-1)
    sims = img_norm @ concept_siglip.T
    top = sims.topk(k, dim=1)
    weights = top.values.clamp(min=0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Per-image blended Y
    picks = concept_llm[top.indices]
    Y = (weights.unsqueeze(-1) * picks).sum(dim=1)

    # Word-balance by primary (top-1) concept
    primary = [kept_concepts[i] for i in top.indices[:, 0].tolist()]
    seen: Counter[str] = Counter()
    keep: list[int] = []
    for i, w in enumerate(primary):
        if seen[w] < max_per_word:
            keep.append(i)
            seen[w] += 1
    print(f"  word-balanced: {N} -> {len(keep)} images (cap {max_per_word}/word, {len(seen)} unique primaries)")
    top5 = seen.most_common(5)
    print(f"  top primary concepts: {top5}")

    X_bal = X_mean[keep].contiguous()
    Y_bal = Y[keep].contiguous()
    del X_mean, Y, img_pooled, img_norm, sims, picks, concept_siglip, concept_llm
    _free()

    # lstsq on the small balanced set (a few thousand × 1152)
    print(f"  solving lstsq on X={tuple(X_bal.shape)} Y={tuple(Y_bal.shape)}...")
    sol = torch.linalg.lstsq(X_bal, Y_bal)
    W = sol.solution
    resid = (X_bal @ W - Y_bal).pow(2).mean().sqrt().item()
    print(f"  W shape: {tuple(W.shape)}  rmse: {resid:.4f}")
    return W
