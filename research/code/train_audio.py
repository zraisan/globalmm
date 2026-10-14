"""Train an audio connector: frozen Whisper-large-v3-turbo encoder -> frozen Llama-3.2-1B-Instruct.

The connector has exactly llama.cpp's ULTRAVOX projector shape (stack 8 frames -> RMSNorm -> Linear ->
SwiGLU -> RMSNorm -> Linear), so it exports into an mmproj and runs in llama.cpp unchanged.
Every clip is padded to Whisper's 30 s window (as llama.cpp does): 1500 frames -> 188 audio tokens.

Tasks (next-token loss on the answer only):
  speech  - LibriSpeech train-clean-100: "Transcribe the audio."
  sounds  - ESC-50 folds 1-4 placed into synthetic tracks of 1-3 non-overlapping events:
            "What sound is this?" / "When can you hear the X? Answer with timestamps." /
            "List the sounds in this audio with their timestamps."   timestamps are [s.s s, e.e s]
Eval: LibriSpeech test-clean WER, ESC-50 fold 5 accuracy, temporal IoU on fold-5 mixtures,
each with a shuffled-audio control.

usage: train_audio.py train [--steps 8000] [--tag a1]
       train_audio.py eval  [--tag a1] [--n 300]
"""
import argparse
import io
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

import train_connector as T

ROOT = T.ROOT
AUD = ROOT / "audio"
WHISPER = AUD / "whisper"
OUT = ROOT / "analysis/audio"
MARKER = "<__media__>"
Q_ASR = "Transcribe the audio."
Q_WHAT = "What sound is this?"
Q_WHEN = "When can you hear the {}? Answer with timestamps."
Q_LIST = "List the sounds in this audio with their timestamps."
SR, MAXLEN = 16000, 29.5


# ---------------------------------------------------------------- model
class AudioConnector(nn.Module):
    """llama.cpp ULTRAVOX projector: mm.a.norm_pre, mm.a.mlp.1, swiglu(silu on 2nd half), mm.a.norm_mid, mm.a.mlp.2"""
    def __init__(self, d_in=1280 * 8, d_hid=4096, d_out=2048):
        super().__init__()
        self.ln_pre = nn.Parameter(torch.ones(d_in))
        self.fc1 = nn.Linear(d_in, d_hid, bias=False)
        self.ln_mid = nn.Parameter(torch.ones(d_hid // 2))
        self.fc2 = nn.Linear(d_hid // 2, d_out, bias=False)

    @staticmethod
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, x):
        h = self.fc1(self.rms(x) * self.ln_pre)
        a, g = h.chunk(2, dim=-1)
        return self.fc2(self.rms(a * F.silu(g)) * self.ln_mid)


def load_whisper():
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder
    cfg = WhisperConfig.from_pretrained(WHISPER)
    with torch.device("cuda"):
        enc = WhisperEncoder(cfg)
    own = enc.state_dict()
    with torch.no_grad(), safe_open(WHISPER / "model.safetensors", framework="pt", device="cuda") as f:
        n = 0
        for k in f.keys():
            if k.startswith("model.encoder.") and k[14:] in own:
                own[k[14:]].copy_(f.get_tensor(k)); n += 1
    assert n == len(own), (n, len(own))
    enc = enc.to(torch.bfloat16).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, WhisperFeatureExtractor.from_pretrained(WHISPER)


@torch.no_grad()
def encode(enc, fe, wavs):
    """list of float32 16 kHz arrays -> [B, 188, 10240] stacked whisper states (float32)"""
    feats = fe(wavs, sampling_rate=SR, return_tensors="pt", device="cuda").input_features
    h = enc(feats.to("cuda", torch.bfloat16)).last_hidden_state.float()  # [B, 1500, 1280]
    h = F.pad(h, (0, 0, 0, 4))                                            # 1504 = 188 * 8
    return h.reshape(h.shape[0], 188, 8 * h.shape[2])


# ---------------------------------------------------------------- data
def sentence(t):
    t = t.strip().lower()
    return t[:1].upper() + t[1:] + "."


class LibriStream:
    def __init__(self, files, seed):
        self.files, self.rng = list(files), random.Random(seed)

    def __iter__(self):
        while True:
            self.rng.shuffle(self.files)
            for fn in self.files:
                pf = pq.ParquetFile(fn)
                groups = list(range(pf.num_row_groups)); self.rng.shuffle(groups)
                for g in groups:
                    t = pf.read_row_group(g, columns=["audio", "text"])
                    aud, txt = t.column("audio").to_pylist(), t.column("text").to_pylist()
                    order = list(range(len(txt))); self.rng.shuffle(order)
                    for i in order:
                        x, sr = sf.read(io.BytesIO(aud[i]["bytes"]), dtype="float32")
                        if sr == SR and len(x) <= MAXLEN * SR:
                            yield x, txt[i]


def libri_eval(n):
    pf, out = pq.ParquetFile(AUD / "ls/test.parquet"), []
    for g in range(pf.num_row_groups):  # one row group at a time: the whole table is 350 MB of flac
        t = pf.read_row_group(g, columns=["audio", "text", "id"])
        for a, txt, i in zip(t.column("audio").to_pylist(), t.column("text").to_pylist(), t.column("id").to_pylist()):
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if len(x) <= MAXLEN * SR:
                out.append((x, txt, i))
            if len(out) >= n:
                return out
    return out


class Esc:
    def __init__(self):
        z = np.load(AUD / "esc50_meta.npz")
        self.wav = np.load(AUD / "esc50_wav.npy", mmap_mode="r")  # 320 MB: read from disk on demand
        self.fold, self.label, self.on, self.off = z["fold"], z["label"], z["on"], z["off"]

    def pool(self, folds):
        return [i for i in range(len(self.fold)) if self.fold[i] in folds]

    def mixture(self, rng, pool, k=None):
        """1-3 non-overlapping events in a 6-25 s track; returns (wav, T, events[(label, s, e)])"""
        k = k or rng.choice([1, 1, 2, 2, 3])
        picks, labs = [], set()
        while len(picks) < k:
            i = rng.choice(pool)
            if self.label[i] not in labs:
                picks.append(i); labs.add(self.label[i])
        Tlen = rng.uniform(5 * k + 1, min(25, 5 * k + 10))
        cuts = sorted(rng.uniform(0, Tlen - 5 * k) for _ in range(k))
        x = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 10 ** rng.uniform(-3.5, -2.5),
                                                                 int(Tlen * SR)).astype(np.float32)
        ev = []
        for j, (i, c) in enumerate(zip(picks, cuts)):
            s = c + 5 * j
            clip = self.wav[i].astype(np.float32)
            peak = np.abs(clip).max() or 1.0
            clip = clip / peak * 10 ** (rng.uniform(-18, -3) / 20)
            a = int(s * SR)
            x[a:a + len(clip)] += clip[: len(x) - a]
            ev.append((str(self.label[i]), round(s + self.on[i], 1), round(s + self.off[i], 1)))
        return x, Tlen, ev


def ts(s, e):
    return f"[{s:.1f}s, {e:.1f}s]"


def esc_task(esc, rng, pool):
    x, _, ev = esc.mixture(rng, pool)
    r = rng.random()
    if len(ev) == 1 and r < 0.5:
        return x, Q_WHAT, ev[0][0].capitalize() + "."
    if r < 0.75:
        lab, s, e = rng.choice(ev)
        return x, Q_WHEN.format(lab), ts(s, e)
    return x, Q_LIST, "\n".join(f"{lab} {ts(s, e)}" for lab, s, e in ev)


def parse_ts(text):
    return [(float(a), float(b)) for a, b in re.findall(r"\[(\d+(?:\.\d+)?)s?,\s*(\d+(?:\.\d+)?)s?\]", text)]


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- prompt
class Prompt:
    def __init__(self, tok):
        self.tok = tok
        self.eot = tok.convert_tokens_to_ids("<|eot_id|>")

    def parts(self, q):
        text = self.tok.apply_chat_template([{"role": "user", "content": MARKER + q}], tokenize=False,
                                            add_generation_prompt=True)
        pre, post = text.split(MARKER)
        return self.tok(pre, add_special_tokens=False).input_ids, self.tok(post, add_special_tokens=False).input_ids


# ---------------------------------------------------------------- train
def train(a):
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt.pt"
    tok, llm = T.load_llm("cuda")
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    enc, fe = load_whisper()
    P = Prompt(tok)
    esc = Esc()
    pool = esc.pool({1, 2, 3, 4})
    files = sorted((AUD / "ls").glob("train100_*.parquet"))
    conn = AudioConnector(d_out=llm.config.hidden_size).cuda()
    opt = torch.optim.AdamW(conn.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    step = 0
    if ck.exists():
        s = torch.load(ck, map_location="cuda")
        conn.load_state_dict(s["conn"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed at step {step}", flush=True)
    else:  # scale the output to the size of Llama's word embeddings, measured on real audio
        rng0 = random.Random(0)
        with torch.no_grad():
            x = encode(enc, fe, [esc.mixture(rng0, pool)[0] for _ in range(4)])
            emb_norm = llm.get_input_embeddings().weight.float().norm(dim=-1).mean()
            conn.fc2.weight.mul_(emb_norm / conn(x).norm(dim=-1).mean())
        print(f"init: target norm {emb_norm:.3f}", flush=True)
    ls = iter(LibriStream(files, seed=step))
    rng = random.Random(1000 + step)
    sched = lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    t0, run, start = time.time(), [], step
    while step < a.steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            batch = []
            for _ in range(a.bs):
                if rng.random() < a.speech:
                    x, txt = next(ls)
                    batch.append((x, Q_ASR, sentence(txt)))
                else:
                    batch.append(esc_task(esc, rng, pool))
            feats = encode(enc, fe, [b[0] for b in batch])
            items = []
            for (_, q, ans), f in zip(batch, feats):
                pre, post = P.parts(q)
                items.append((pre, f, post, tok(ans, add_special_tokens=False).input_ids + [P.eot]))
            emb, att, lab = T.embed_batch(llm, conn, items, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hid = llm.model(inputs_embeds=emb, attention_mask=att).last_hidden_state
                tgt = lab[:, 1:]
                sel = tgt != -100
                loss = F.cross_entropy(llm.lm_head(hid[:, :-1][sel]).float(), tgt[sel]) / a.accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(conn.parameters(), 1.0)
        opt.step()
        run.append(tot)
        if step % 50 == 0:
            rate = (time.time() - t0) / (step - start)
            print(f"step {step}/{a.steps}  loss {np.mean(run[-50:]):.3f}  {rate:.2f}s/step  "
                  f"eta {(a.steps - step) * rate / 3600:.1f}h  vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                  flush=True)
        if step % 100 == 0 or step == a.steps:  # often: the watchdog may stop us when the PC is busy
            tmp = out / "ckpt.tmp"
            torch.save(dict(conn=conn.state_dict(), opt=opt.state_dict(), step=step), tmp)
            tmp.rename(ck)
            torch.save(conn.state_dict(), out / "connector.pt")
            json.dump(dict(step=step, loss_tail=run[-500:], args=vars(a)), open(out / "train_log.json", "w"))
    print("DONE", flush=True)


# ---------------------------------------------------------------- eval
def norm_text(t):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


@torch.no_grad()
def evaluate(a):
    import jiwer
    tok, llm = T.load_llm("cuda")
    enc, fe = load_whisper()
    P = Prompt(tok)
    conn = AudioConnector(d_out=llm.config.hidden_size).cuda()
    conn.load_state_dict(torch.load(OUT / a.tag / "connector.pt"))

    def gen(x, q, n):
        pre, post = P.parts(q)
        emb, att, _ = T.embed_batch(llm, conn, [(pre, encode(enc, fe, [x])[0], post, [])], "cuda")
        o = llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
        return tok.decode(o[0], skip_special_tokens=True).strip()

    res, raw = {}, {}
    # speech
    ls = libri_eval(a.n)
    for cond in ("audio", "shuffled"):
        hyp, ref, rows = [], [], []
        for k, (x, txt, i) in enumerate(ls):
            src = x if cond == "audio" else ls[(k + 7) % len(ls)][0]
            h = gen(src, Q_ASR, 120)
            hyp.append(norm_text(h) or "-"); ref.append(norm_text(txt)); rows.append(dict(id=i, ref=txt, hyp=h))
        res[f"asr_wer_{cond}"] = jiwer.wer(ref, hyp)
        raw[f"asr_{cond}"] = rows
        print(f"ASR {cond}: WER {res[f'asr_wer_{cond}']:.3f}", flush=True)
    # sound classification (fold 5, single clip at a random offset)
    esc = Esc()
    pool = esc.pool({5})
    rng = random.Random(5)
    mixes1 = [esc.mixture(rng, pool, k=1) for _ in range(a.n)]
    for cond in ("audio", "shuffled"):
        rows = []
        for k, (x, _, ev) in enumerate(mixes1):
            src = x if cond == "audio" else mixes1[(k + 7) % len(mixes1)][0]
            h = gen(src, Q_WHAT, 20)
            rows.append(dict(gt=ev[0][0], hyp=h, ok=norm_text(h) == norm_text(ev[0][0])))
        res[f"esc_acc_{cond}"] = float(np.mean([r["ok"] for r in rows]))
        raw[f"esc_{cond}"] = rows
        print(f"ESC {cond}: acc {res[f'esc_acc_{cond}']:.3f}", flush=True)
    # temporal grounding (fold 5 mixtures of 1-3 events)
    rng = random.Random(55)
    mixes = [esc.mixture(rng, pool) for _ in range(a.n)]
    qs = [rng.choice(ev) for _, _, ev in mixes]
    for cond in ("audio", "shuffled"):
        rows = []
        for k, ((x, Tlen, ev), q) in enumerate(zip(mixes, qs)):
            src = x if cond == "audio" else mixes[(k + 7) % len(mixes)][0]
            h = gen(src, Q_WHEN.format(q[0]), 30)
            p = parse_ts(h)
            lst = gen(src, Q_LIST, 80)
            said = {}
            for ln in lst.splitlines():
                pt = parse_ts(ln)
                if pt:
                    said[norm_text(re.sub(r"\s*\[.*", "", ln))] = pt[0]
            gtl = {norm_text(l): (s, e) for l, s, e in ev}
            hit = [l for l in said if l in gtl]
            rows.append(dict(T=Tlen, events=ev, q=q[0], when=h, iou=tiou(p[0], q[1:]) if p else 0.0,
                             whole_iou=tiou((0, Tlen), q[1:]), list=lst,
                             cls_p=len(hit) / max(1, len(said)), cls_r=len(hit) / len(gtl),
                             seg_r=sum(tiou(said[l], gtl[l]) >= 0.5 for l in hit) / len(gtl)))
        ious = np.array([r["iou"] for r in rows])
        res[f"when_iou_{cond}"] = float(ious.mean()); res[f"when_acc50_{cond}"] = float((ious >= .5).mean())
        for m in ("cls_p", "cls_r", "seg_r"):
            res[f"list_{m}_{cond}"] = float(np.mean([r[m] for r in rows]))
        if cond == "audio":
            w = np.array([r["whole_iou"] for r in rows])
            res["when_iou_wholetrack"] = float(w.mean()); res["when_acc50_wholetrack"] = float((w >= .5).mean())
        raw[f"when_{cond}"] = rows
        print(f"WHEN {cond}: IoU {ious.mean():.3f} acc@0.5 {(ious >= .5).mean():.3f}", flush=True)
    print(json.dumps(res, indent=1))
    json.dump(dict(summary=res, raw=raw), open(OUT / a.tag / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--speech", type=float, default=0.5)
    p.add_argument("--tag", default="a1")
    p.add_argument("--n", type=int, default=300)
    a = p.parse_args()
    {"train": train, "eval": evaluate}[a.cmd](a)
