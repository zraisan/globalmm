"""ESC-50 -> 16 kHz float16 arrays + active extent per clip (for timestamp ground truth).

Writes audio/esc50_wav.npy [2000, 80000] f16 and audio/esc50_meta.npz: fold, label (str), on/off (seconds).
Active extent: 50 ms RMS frames above 10% of the clip's peak RMS (-20 dB); first..last active frame.
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "audio/esc50/ESC-50-master"
rows = list(csv.DictReader(open(E / "meta/esc50.csv")))
wav = np.zeros((len(rows), 80000), np.float16)
on, off = np.zeros(len(rows)), np.zeros(len(rows))
for i, r in enumerate(rows):
    x, sr = sf.read(E / "audio" / r["filename"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    y = resample_poly(x, 160, 441)[:80000]
    wav[i, :len(y)] = y
    rms = np.sqrt(np.mean(y[: len(y) // 800 * 800].reshape(-1, 800) ** 2, 1))
    act = np.nonzero(rms > 0.1 * rms.max())[0] if rms.max() > 0 else np.array([0])
    on[i], off[i] = act[0] * 0.05, (act[-1] + 1) * 0.05
np.save(ROOT / "audio/esc50_wav.npy", wav)  # memory-mapped by train_audio.py
np.savez(ROOT / "audio/esc50_meta.npz", fold=np.array([int(r["fold"]) for r in rows]),
         label=np.array([r["category"].replace("_", " ") for r in rows]), on=on, off=off,
         name=np.array([r["filename"] for r in rows]))
print(len(rows), "clips; mean active", np.mean(off - on).round(2), "s; full-length", np.mean(off - on > 4.9).round(2))
