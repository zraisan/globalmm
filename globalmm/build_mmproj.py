"""
Build a llama.cpp mmproj.gguf file that bundles SigLIP + our W projector.

The file uses projector_type=gemma3 because that's the only existing llama.cpp
projector that takes SigLIP features and applies a single linear projection
to reach the LLM's embedding space. Gemma3's graph:

    patches -> avg_pool(k=n_merge) -> rms_norm -> * soft_emb_norm_w -> @ mm_input_proj_w

We set soft_emb_norm_w = all ones (identity multiply), and mm_input_proj_w = our W.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import SiglipModel, SiglipProcessor
import gguf


SIGLIP_MODEL = "google/siglip-so400m-patch14-384"

# HF SigLIP name -> llama.cpp v.* name template (with {i} for block index)
# Shape transform is applied inside _copy_tensor.
_BLOCK_MAP = {
    "encoder.layers.{i}.layer_norm1.weight":       "v.blk.{i}.ln1.weight",
    "encoder.layers.{i}.layer_norm1.bias":         "v.blk.{i}.ln1.bias",
    "encoder.layers.{i}.layer_norm2.weight":       "v.blk.{i}.ln2.weight",
    "encoder.layers.{i}.layer_norm2.bias":         "v.blk.{i}.ln2.bias",
    "encoder.layers.{i}.self_attn.q_proj.weight":  "v.blk.{i}.attn_q.weight",
    "encoder.layers.{i}.self_attn.q_proj.bias":    "v.blk.{i}.attn_q.bias",
    "encoder.layers.{i}.self_attn.k_proj.weight":  "v.blk.{i}.attn_k.weight",
    "encoder.layers.{i}.self_attn.k_proj.bias":    "v.blk.{i}.attn_k.bias",
    "encoder.layers.{i}.self_attn.v_proj.weight":  "v.blk.{i}.attn_v.weight",
    "encoder.layers.{i}.self_attn.v_proj.bias":    "v.blk.{i}.attn_v.bias",
    "encoder.layers.{i}.self_attn.out_proj.weight":"v.blk.{i}.attn_out.weight",
    "encoder.layers.{i}.self_attn.out_proj.bias":  "v.blk.{i}.attn_out.bias",
    "encoder.layers.{i}.mlp.fc1.weight":           "v.blk.{i}.ffn_up.weight",
    "encoder.layers.{i}.mlp.fc1.bias":             "v.blk.{i}.ffn_up.bias",
    "encoder.layers.{i}.mlp.fc2.weight":           "v.blk.{i}.ffn_down.weight",
    "encoder.layers.{i}.mlp.fc2.bias":             "v.blk.{i}.ffn_down.bias",
}


def _to_numpy(t: torch.Tensor, target_dtype: np.dtype) -> np.ndarray:
    """Convert a torch tensor to a numpy array of the given dtype."""
    t = t.detach().cpu()
    if target_dtype == np.float16:
        t = t.half()
    elif target_dtype == np.float32:
        t = t.float()
    return t.numpy().astype(target_dtype)


def build_mmproj(
    W_projector: torch.Tensor,
    output_path: str | Path,
    siglip_name: str = SIGLIP_MODEL,
    n_merge: int = 3,
    weight_dtype: np.dtype = np.float16,
):
    """
    Build a llama.cpp mmproj.gguf file that bundles SigLIP + the given projector.

    Args:
        W_projector: (siglip_embd, llm_embd) projector matrix, the W we compute via lstsq.
                     SigLIP's embed dim is 1152 for SO400M. llm_embd is the user LLM's hidden size.
        output_path: where to write the .gguf file.
        siglip_name: HF model name for the vision encoder.
        n_merge: spatial pooling factor for the gemma3 projector (3 -> 27x27 to 9x9 = 81 tokens).
        weight_dtype: np.float16 or np.float32.
    """
    output_path = Path(output_path)

    # ── Load SigLIP ─────────────────────────────────────────────────────────
    print(f"Loading {siglip_name}...")
    model = SiglipModel.from_pretrained(siglip_name)
    processor = SiglipProcessor.from_pretrained(siglip_name)
    vision = model.vision_model
    config = model.config.vision_config

    siglip_embd     = config.hidden_size       # 1152
    siglip_ffn      = config.intermediate_size # 4304
    siglip_layers   = config.num_hidden_layers # 27
    siglip_heads    = config.num_attention_heads # 16
    image_size      = config.image_size        # 384
    patch_size      = config.patch_size        # 14
    layernorm_eps   = config.layer_norm_eps    # 1e-6
    n_positions     = (image_size // patch_size) ** 2  # 729

    # Preprocessor stats
    img_mean = list(processor.image_processor.image_mean)  # [0.5, 0.5, 0.5]
    img_std  = list(processor.image_processor.image_std)   # [0.5, 0.5, 0.5]

    # LLM embed dim from W shape
    assert W_projector.shape[0] == siglip_embd, \
        f"W has wrong input dim: expected {siglip_embd}, got {W_projector.shape[0]}"
    llm_embd = W_projector.shape[1]

    print(f"  SigLIP: {siglip_embd}-dim, {siglip_layers} layers, image {image_size}x{image_size}")
    print(f"  Projector: {siglip_embd} -> {llm_embd}, n_merge={n_merge}")

    # ── Create gguf writer ──────────────────────────────────────────────────
    writer = gguf.GGUFWriter(str(output_path), "clip")

    # ── Metadata ────────────────────────────────────────────────────────────
    writer.add_string("general.architecture", "clip")
    writer.add_string("clip.projector_type", "gemma3")
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_llava_projector", False)

    writer.add_uint32("clip.vision.image_size", image_size)
    writer.add_uint32("clip.vision.patch_size", patch_size)
    writer.add_uint32("clip.vision.embedding_length", siglip_embd)
    writer.add_uint32("clip.vision.feed_forward_length", siglip_ffn)
    writer.add_uint32("clip.vision.projection_dim", llm_embd)
    writer.add_uint32("clip.vision.block_count", siglip_layers)
    writer.add_uint32("clip.vision.attention.head_count", siglip_heads)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", layernorm_eps)
    writer.add_array("clip.vision.image_mean", img_mean)
    writer.add_array("clip.vision.image_std", img_std)
    writer.add_bool("clip.use_gelu", True)

    # projector scale factor — non-default (gemma3 default = 4)
    if n_merge != 4:
        writer.add_uint32("clip.vision.projector.scale_factor", n_merge)

    # ── Vision tensors: patch embedding ─────────────────────────────────────
    # gguf-py writer REVERSES the numpy shape when storing dims, so we pass
    # HF's native shapes and end up with the llama.cpp-expected ggml ne[] order.
    state = vision.state_dict()

    # patch_embedding: HF shape (1152, 3, 14, 14), pass as-is -> file dims
    # [14, 14, 3, 1152] = ggml ne (KW, KH, C_in, C_out) ✓
    pw = state["embeddings.patch_embedding.weight"]
    writer.add_tensor("v.patch_embd.weight", _to_numpy(pw, np.float32))
    writer.add_tensor("v.patch_embd.bias",   _to_numpy(state["embeddings.patch_embedding.bias"], np.float32))

    # position_embedding: HF shape (729, 1152), pass as-is -> file dims
    # [1152, 729] = ggml ne (n_embd, n_pos) ✓
    pe = state["embeddings.position_embedding.weight"]
    writer.add_tensor("v.position_embd.weight", _to_numpy(pe, np.float32))

    # ── Vision tensors: encoder blocks ──────────────────────────────────────
    for i in range(siglip_layers):
        for hf_tmpl, gg_tmpl in _BLOCK_MAP.items():
            hf_name = hf_tmpl.format(i=i)
            gg_name = gg_tmpl.format(i=i)
            t = state[hf_name]
            # Most tensors: same shape. No transform.
            # (Linear .weight is stored as (out, in) in both PyTorch and llama.cpp.)
            # Norm tensors: 1D, no transform.
            dtype = weight_dtype if t.dim() == 2 else np.float32
            writer.add_tensor(gg_name, _to_numpy(t, dtype))

    # Post layernorm
    writer.add_tensor("v.post_ln.weight", _to_numpy(state["post_layernorm.weight"], np.float32))
    writer.add_tensor("v.post_ln.bias",   _to_numpy(state["post_layernorm.bias"],   np.float32))

    # ── Projector tensors ───────────────────────────────────────────────────
    # Real gemma3 reader-shape for mm.input_projection.weight is (projection_dim, n_embd).
    # Writer reverses input numpy shape, so we need to pass numpy (n_embd, projection_dim).
    # Our W from lstsq already has shape (siglip_embd=1152, llm_embd=projection_dim). Pass as-is.
    writer.add_tensor("mm.input_projection.weight", _to_numpy(W_projector, weight_dtype))

    # soft_emb_norm: all ones (identity element-wise multiply)
    soft_norm = torch.ones(siglip_embd, dtype=torch.float32)
    writer.add_tensor("mm.soft_emb_norm.weight", _to_numpy(soft_norm, np.float32))

    # ── Write ───────────────────────────────────────────────────────────────
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")
