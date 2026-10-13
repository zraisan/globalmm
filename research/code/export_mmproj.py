"""Write a trained connector into the Qwen3.8 mmproj (replaces mm.0 / mm.2, sets projection_dim).

usage: export_mmproj.py CONNECTOR_PT MMPROJ_IN OUT_GGUF
"""
import sys

import gguf
import numpy as np
import torch

ckpt, src, out = sys.argv[1:4]
sd = torch.load(ckpt, map_location="cpu")
new = {
    "mm.0.weight": sd["fc1.weight"].float().numpy(), "mm.0.bias": sd["fc1.bias"].float().numpy(),
    "mm.2.weight": sd["fc2.weight"].float().numpy(), "mm.2.bias": sd["fc2.bias"].float().numpy(),
}
d_out = new["mm.2.weight"].shape[0]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(out, "clip")
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, d_out if f.name == "clip.vision.projection_dim" else f.contents(), vt, sub_type=st)
datas = []
for t in r.tensors:
    if t.name in new:
        a = np.ascontiguousarray(new[t.name].astype(np.float32))
        w.add_tensor_info(t.name, a.shape, a.dtype, a.nbytes, gguf.GGMLQuantizationType.F32)
        datas.append(a)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        datas.append(t.data)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for a in datas:
    w.write_tensor_data(a, tensor_endianess=r.endianess)
w.close()
print("wrote", out, "projection_dim", d_out)
