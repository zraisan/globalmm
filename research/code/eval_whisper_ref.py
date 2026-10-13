"""Reference point for speech: Whisper-large-v3-turbo with its own decoder on the same LibriSpeech
test-clean utterances as train_audio.py eval (same text normalization, same WER).

usage: eval_whisper_ref.py [N]   -> analysis/audio/whisper_ref.json
"""
import json
import sys

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import train_audio as A


@torch.no_grad()
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    proc = WhisperProcessor.from_pretrained(A.WHISPER)
    model = WhisperForConditionalGeneration.from_pretrained(A.WHISPER, dtype=torch.bfloat16, device_map="cuda").eval()
    ref, hyp, rows = [], [], []
    for x, txt, i in A.libri_eval(n):
        f = proc(x, sampling_rate=A.SR, return_tensors="pt").input_features.to("cuda", torch.bfloat16)
        o = model.generate(f, language="en", task="transcribe", max_new_tokens=200)
        h = proc.batch_decode(o, skip_special_tokens=True)[0].strip()
        ref.append(A.norm_text(txt)); hyp.append(A.norm_text(h) or "-"); rows.append(dict(id=i, ref=txt, hyp=h))
    w = jiwer.wer(ref, hyp)
    print(f"Whisper-large-v3-turbo WER {w:.4f} on {len(ref)} utterances", flush=True)
    A.OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(wer=w, n=len(ref), rows=rows), open(A.OUT / "whisper_ref.json", "w"), indent=1)


if __name__ == "__main__":
    main()
