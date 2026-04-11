"""globalmm: build llama.cpp mmproj.gguf files for any LLM via lstsq."""

from .projector import compute_W
from .build_mmproj import build_mmproj
from .encode import ensure_caches, ensure_images, encode_dataset

__all__ = ["compute_W", "build_mmproj", "ensure_caches", "ensure_images", "encode_dataset"]
