"""Export Whisper-large-v3-turbo encoder + trained audio connector as a llama.cpp mmproj (ULTRAVOX type).

Builds a minimal Ultravox-style HF folder (audio_tower.* + multi_modal_projector.*) and runs llama.cpp's
own converter on it, so the GGUF layout is exactly what llama.cpp expects.

usage: export_audio_mmproj.py CONNECTOR.pt OUT.gguf
"""
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "audio/whisper"
conn_pt, out = Path(sys.argv[1]), Path(sys.argv[2])
hf = conn_pt.parent / "hf"
hf.mkdir(exist_ok=True)

c = torch.load(conn_pt, map_location="cpu")
t = {"multi_modal_projector.ln_pre.weight": c["ln_pre"], "multi_modal_projector.linear_1.weight": c["fc1.weight"],
     "multi_modal_projector.ln_mid.weight": c["ln_mid"], "multi_modal_projector.linear_2.weight": c["fc2.weight"]}
with safe_open(W / "model.safetensors", framework="pt") as f:
    for k in f.keys():
        if k.startswith("model.encoder."):
            t["audio_tower." + k[14:]] = f.get_tensor(k)
save_file({k: v.contiguous() for k, v in t.items()}, hf / "model.safetensors")
wcfg = json.load(open(W / "config.json"))
json.dump(dict(architectures=["UltravoxModel"], model_type="ultravox", stack_factor=8,
               audio_config=wcfg, text_config=dict(hidden_size=int(c["fc2.weight"].shape[0]))),
          open(hf / "config.json", "w"), indent=1)
subprocess.run([sys.executable, str(ROOT / "llama.cpp/convert_hf_to_gguf.py"), str(hf), "--mmproj",
                "--outtype", "f16", "--outfile", str(out)], check=True)
print("wrote", out)
