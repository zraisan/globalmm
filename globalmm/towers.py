"""
Load the vision tower of an existing llama.cpp mmproj.gguf into PyTorch, so a new projector can be
trained on its features, and write a new mmproj with the trained projector in place of the old one.

The tower weights are read from the GGUF file itself (dequantized), so the original Hugging Face
model is not needed and training sees the same weights llama.cpp uses at inference.

Each supported `clip.projector_type` has an adapter. An adapter knows how to:
  - rebuild the tower as a transformers module and load the GGUF tensors into it;
  - preprocess an image the way the tower expects;
  - return the projector's input features, one row per LLM token;
  - name the projector tensors that get replaced on export.
"""

from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _field(reader: gguf.GGUFReader, name: str, default=None):
    f = reader.fields.get(name)
    return default if f is None else f.contents()


def _tensor(t) -> torch.Tensor:
    """GGUF tensor -> float32 torch tensor in PyTorch layout (ggml stores dims reversed)."""
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
        a = np.asarray(t.data)
    else:
        a = gguf.quants.dequantize(t.data, t.tensor_type)
    shape = tuple(int(v) for v in reversed(t.shape))
    return torch.from_numpy(np.array(a, dtype=np.float32).reshape(shape))  # copy: GGUF data is read-only


@dataclass
class ProjectorSpec:
    d_in: int        # width of one projector input row
    d_hidden: int    # width of the projector's hidden layer
    act: str         # activation between the two layers
    names: tuple     # GGUF names (fc1.weight, fc1.bias, fc2.weight, fc2.bias)


class QwenMergerTower:
    """`qwen3vl_merger`: Qwen3-VL style vision tower (also used by Qwen3.5-style models).

    The tower ends in a patch merger: LayerNorm (v.post_ln), 2x2 patch grouping, then
    mm.0 -> GELU -> mm.2. We keep the LayerNorm and grouping, and train a replacement for
    mm.0 / mm.2. Deepstack mergers, if present, are dropped on export because a text-only LLM
    cannot consume them.
    """

    projector_types = ("qwen3vl_merger",)
    image_wrap = ("<|vision_start|>", "<|vision_end|>")  # text llama.cpp puts around Qwen-style image embeddings

    def __init__(self, reader: gguf.GGUFReader, device: str = "cuda", dtype=torch.bfloat16):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding

        tensors = {t.name: t for t in reader.tensors}
        hidden = int(_field(reader, "clip.vision.embedding_length"))
        self.patch = int(_field(reader, "clip.vision.patch_size"))
        self.merge = int(_field(reader, "clip.vision.spatial_merge_size", 2))
        self.mean = [float(v) for v in _field(reader, "clip.vision.image_mean")]
        self.std = [float(v) for v in _field(reader, "clip.vision.image_std")]
        n_pos = tensors["v.position_embd.weight"].shape[1]
        cfg = Qwen3VLVisionConfig(
            depth=int(_field(reader, "clip.vision.block_count")),
            hidden_size=hidden,
            intermediate_size=int(_field(reader, "clip.vision.feed_forward_length")),
            num_heads=int(_field(reader, "clip.vision.attention.head_count")),
            patch_size=self.patch,
            spatial_merge_size=self.merge,
            temporal_patch_size=2,
            out_hidden_size=int(_field(reader, "clip.vision.projection_dim")),
            num_position_embeddings=int(n_pos),
            deepstack_visual_indexes=[],
        )
        with torch.device("meta"):
            vm = Qwen3VLVisionModel(cfg)
        sd = {}
        for name in vm.state_dict():
            sd[name] = self._load(name, tensors)
        vm.load_state_dict(sd, assign=True)
        vm.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(cfg)   # buffers are not in the state dict
        self.vm = vm.to(device=device, dtype=dtype).eval()
        for p in self.vm.parameters():
            p.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self._grab = {}
        self.vm.merger.linear_fc1.register_forward_pre_hook(lambda m, a: self._grab.__setitem__("x", a[0].detach()))
        fc1 = tensors["mm.0.weight"]
        self.spec = ProjectorSpec(d_in=int(fc1.shape[0]), d_hidden=int(fc1.shape[1]), act="gelu",
                                  names=("mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"))
        self.init_fc1 = (_tensor(tensors["mm.0.weight"]), _tensor(tensors["mm.0.bias"]))

    @staticmethod
    def _load(name: str, tensors: dict) -> torch.Tensor:
        if name == "patch_embed.proj.weight":   # conv3d split by the converter into two conv2d (t = 0, 1)
            return torch.stack([_tensor(tensors["v.patch_embd.weight"]), _tensor(tensors["v.patch_embd.weight.1"])], 2)
        fixed = {"patch_embed.proj.bias": "v.patch_embd.bias", "pos_embed.weight": "v.position_embd.weight",
                 "merger.norm.weight": "v.post_ln.weight", "merger.norm.bias": "v.post_ln.bias",
                 "merger.linear_fc1.weight": "mm.0.weight", "merger.linear_fc1.bias": "mm.0.bias",
                 "merger.linear_fc2.weight": "mm.2.weight", "merger.linear_fc2.bias": "mm.2.bias"}
        if name in fixed:
            return _tensor(tensors[fixed[name]])
        _, i, rest = name.split(".", 2)       # blocks.<i>.<rest>
        part = {"attn.qkv": "attn_qkv", "attn.proj": "attn_out", "norm1": "ln1", "norm2": "ln2",
                "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}
        mod, kind = rest.rsplit(".", 1)
        return _tensor(tensors[f"v.blk.{i}.{part[mod]}.{kind}"])

    def fit(self, img: Image.Image, longest: int = 448) -> Image.Image:
        """Resize so the longest side is about `longest` and both sides are multiples of patch*merge."""
        unit = self.patch * self.merge
        s = longest / max(img.size)
        return img.resize(tuple(max(unit, round(v * s / unit) * unit) for v in img.size), Image.BICUBIC)

    def preprocess(self, img: Image.Image):
        """Image already sized to multiples of patch*merge -> (pixel rows, grid_thw), in HF Qwen layout."""
        p, m = self.patch, self.merge
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
        x = (x - torch.tensor(self.mean)) / torch.tensor(self.std)
        x = x.permute(2, 0, 1)                                     # C, H, W
        C, H, W = x.shape
        gh, gw = H // p, W // p
        x = x.unsqueeze(0).repeat(2, 1, 1, 1)                      # temporal patch of 2 identical frames
        x = x.reshape(1, 2, C, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)                   # t, bh, bw, mh, mw, C, tp, ph, pw
        return x.reshape(gh * gw, C * 2 * p * p), torch.tensor([[1, gh, gw]])

    @torch.no_grad()
    def features(self, batch):
        """list of (pixel rows, grid_thw) -> list of projector-input tensors [n_tokens, d_in] (float32)."""
        pv = torch.cat([b[0] for b in batch]).to(self.device, self.dtype)
        thw = torch.cat([b[1] for b in batch]).to(self.device)
        self.vm(pv, grid_thw=thw)
        sizes = [int(t * h * w) // (self.merge ** 2) for t, h, w in thw.tolist()]
        return list(self._grab["x"].float().split(sizes))

    @staticmethod
    def drop_on_export(name: str) -> bool:
        return name.startswith("v.deepstack.")

    @staticmethod
    def export_fields(name: str, value, d_out: int):
        if name == "clip.vision.projection_dim":
            return d_out
        if name == "clip.vision.is_deepstack_layers":
            return [False] * len(value)
        return value


ADAPTERS = {pt: cls for cls in (QwenMergerTower,) for pt in cls.projector_types}


def open_tower(mmproj: Path, device: str = "cuda"):
    reader = gguf.GGUFReader(str(mmproj))
    ptype = _field(reader, "clip.projector_type")
    if ptype not in ADAPTERS:
        raise SystemExit(f"projector type {ptype!r} is not supported for --method connector yet. "
                         f"Supported: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[ptype](reader, device=device), reader


class Projector(nn.Module):
    """The trained replacement: same shape as the mmproj's own two-layer projector, new output width."""

    def __init__(self, spec: ProjectorSpec, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(spec.d_in, spec.d_hidden)
        self.fc2 = nn.Linear(spec.d_hidden, d_out)

    def forward(self, x):
        return self.fc2(nn.functional.gelu(self.fc1(x), approximate="tanh"))


def export_mmproj(reader: gguf.GGUFReader, tower, proj: Projector, out: Path) -> None:
    """Copy the source mmproj, replacing the projector tensors and the projection width."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in proj.state_dict().items()}
    n = tower.spec.names
    new = {n[0]: sd["fc1.weight"], n[1]: sd["fc1.bias"], n[2]: sd["fc2.weight"], n[3]: sd["fc2.bias"]}
    d_out = sd["fc2.weight"].shape[0]
    w = gguf.GGUFWriter(str(out), "clip")
    for f in reader.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, tower.export_fields(f.name, f.contents(), d_out), vt, sub_type=st)
    datas = []
    for t in reader.tensors:
        if tower.drop_on_export(t.name):
            continue
        if t.name in new:
            a = np.ascontiguousarray(new[t.name].astype(np.float32))
            w.add_tensor_info(t.name, a.shape, a.dtype, a.nbytes, gguf.GGMLQuantizationType.F32)
            datas.append(a)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            datas.append(t.data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for a in datas:
        w.write_tensor_data(a, tensor_endianess=reader.endianess)
    w.close()
