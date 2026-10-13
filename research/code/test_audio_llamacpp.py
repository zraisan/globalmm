"""Audio test through llama-server (text-only Llama GGUF + our audio mmproj), on held-out clips:
LibriSpeech test-clean utterances, single ESC-50 fold-5 sounds, and fold-5 mixtures (timestamps).
Also writes the WAVs so they can be listened to.

usage: test_audio_llamacpp.py PORT OUT_DIR
"""
import base64
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

import train_audio as A

port, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
url = f"http://127.0.0.1:{port}"
for _ in range(300):
    try:
        if requests.get(url + "/health", timeout=2).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    sys.exit("server not healthy")


def ask(x, q, n=120):
    buf = io.BytesIO(); sf.write(buf, x, A.SR, format="WAV", subtype="PCM_16")
    r = requests.post(url + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}},
            {"type": "text", "text": q}]}],
        "max_tokens": n, "temperature": 0, "seed": 0}, timeout=900).json()
    return r["choices"][0]["message"]["content"].strip()


res = []
for x, txt, i in A.libri_eval(4):
    sf.write(out / f"speech_{i}.wav", x, A.SR)
    res.append(dict(clip=f"speech_{i}", q=A.Q_ASR, truth=txt, answer=ask(x, A.Q_ASR)))
esc, rng = A.Esc(), random.Random(123)
pool = esc.pool({5})
for k in range(3):
    x, T, ev = esc.mixture(rng, pool, k=1)
    sf.write(out / f"sound_{k}.wav", x, A.SR)
    res.append(dict(clip=f"sound_{k}", q=A.Q_WHAT, truth=ev[0][0], answer=ask(x, A.Q_WHAT, 20)))
for k in range(3):
    x, T, ev = esc.mixture(rng, pool, k=3)
    sf.write(out / f"mix_{k}.wav", x, A.SR)
    truth = "\n".join(f"{l} {A.ts(s, e)}" for l, s, e in ev)
    res.append(dict(clip=f"mix_{k}", q=A.Q_LIST, truth=truth, answer=ask(x, A.Q_LIST, 80)))
    l, s, e = ev[1]
    res.append(dict(clip=f"mix_{k}", q=A.Q_WHEN.format(l), truth=A.ts(s, e), answer=ask(x, A.Q_WHEN.format(l), 30)))
for r in res:
    print(f"### {r['clip']}  {r['q']}\n  truth : {r['truth']!r}\n  answer: {r['answer']!r}", flush=True)
json.dump(res, open(out / "results.json", "w"), indent=1, default=float)
