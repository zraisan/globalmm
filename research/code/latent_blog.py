"""Latent-space data for the blog: where image and audio tokens sit relative to word embeddings,
before and after the trained connectors.

Vision (held-out val2017 eval images, tokens labelled by COCO masks):
  q_tok  Qwen3.8 projector output (what Qwen's own LLM reads)          vs Qwen words
  naive  q_tok mapped with the Qwen->Llama word rotation (failed idea)  vs Llama words
  c_tok  our trained connector output (what Llama reads)                vs Llama words
Audio (held-out ESC-50 fold 5 clips + LibriSpeech test-clean):
  w_tok  stacked Whisper states (own space, no words)
  a_tok  our trained audio connector output                             vs Llama words

Writes analysis/latent_blog.json (metrics, 2D maps, per-token heatmaps for the showcase images).
"""
import json
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

import spaces as S
import train_audio as A
import train_connector as T

torch.manual_seed(0)
OUT = T.ROOT / "analysis/latent_blog.json"
SHOW = ["000000000785", "000000000724", "000000000139", "000000039769", "000000000632", "000000001000"]
N_IMG = 80


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


def rank_of(tok, W, target_idx):
    """rank (0 = nearest) of target word among all vocab rows, by cosine"""
    sims = unit(tok) @ unit(W).T
    tgt = sims[np.arange(len(tok)), target_idx]
    return (sims > tgt[:, None]).sum(1)


# ---------------------------------------------------------------- words
qw, lw = S.qwen_words(), S.llama_words()
vocab = S.shared_english(qw, lw)[:20000]
names, _, _ = S.coco_categories()
head = lambda n: n.split()[-1] if n not in ("stop sign", "fire hydrant", "teddy bear", "hot dog") else n.split()[0]
cat_word = [{"tv": "television", "couch": "sofa", "dining table": "table", "potted plant": "plant",
             "cell phone": "phone", "sports ball": "ball", "hair drier": "dryer", "wine glass": "glass",
             "baseball glove": "glove", "tennis racket": "racket", "baseball bat": "bat", "traffic light": "light",
             "parking meter": "meter"}.get(n, head(n)) for n in names]
n_shared = len(vocab)
Wq = qw.rows([qw.vocab["Ġ" + w] for w in vocab])
Wl = lw.rows([lw.vocab["Ġ" + w] for w in vocab])
extra = [w for w in dict.fromkeys(cat_word) if w not in vocab]  # multi-token names: mean of their pieces
vocab += extra
Wq = np.concatenate([Wq, np.stack([qw.word_vec(w) for w in extra])]) if extra else Wq
Wl = np.concatenate([Wl, np.stack([lw.word_vec(w) for w in extra])]) if extra else Wl
vi = {w: i for i, w in enumerate(vocab)}
print("extra multi-token category words:", extra, flush=True)
# Qwen -> Llama: scaled orthogonal Procrustes on the 20k shared single-token words
U, sv, Vt = np.linalg.svd(Wq[:n_shared].T @ Wl[:n_shared], full_matrices=False)
R = U @ Vt
scale = sv.sum() / (Wq[:n_shared] ** 2).sum()
hold = np.arange(len(vocab)) % 10 == 0
fit_top1 = float(np.mean(np.argmax(unit(Wq[hold] @ R) @ unit(Wl).T, 1) == np.nonzero(hold)[0]))
print("procrustes top1 (all words incl. fit set, 1/10 sample):", fit_top1, flush=True)

# ---------------------------------------------------------------- vision tokens
meta = json.load(open(T.DATA / "meta.json"))
by = {m["stem"]: m for m in meta["metas"]}
ev = [by[s] for s in meta["split"]["eval"]]
pick = SHOW + [m["stem"] for m in ev if m["stem"] not in SHOW][:N_IMG - len(SHOW)]
X = np.load(T.DATA / "x.npy", mmap_mode="r")
imgs = [dict(stem=s, gh=by[s]["gh"], gw=by[s]["gw"], W=by[s]["W"], H=by[s]["H"]) for s in pick]
labs = S.token_labels(imgs)
with safe_open(T.QWEN_SHARD, framework="pt") as f:
    g = lambda k: f.get_tensor("model.visual.merger." + k).float()
    q1w, q1b, q2w, q2b = g("linear_fc1.weight"), g("linear_fc1.bias"), g("linear_fc2.weight"), g("linear_fc2.bias")
conn = T.Connector(2048)
conn.load_state_dict(torch.load(T.DATA / "run_full/connector.pt", map_location="cpu"))
Q, C, L, IMG = [], [], [], []
with torch.no_grad():
    for k, s in enumerate(pick):
        m = by[s]
        x = torch.from_numpy(np.asarray(X[m["off"]:m["off"] + m["n"]], dtype=np.float32))
        assert m["n"] == m["gh"] * m["gw"], (m["n"], m["gh"], m["gw"])
        Q.append((F.gelu(x @ q1w.T + q1b) @ q2w.T + q2b).numpy())
        C.append(conn(x).numpy())
        L.append(labs[k]); IMG.append(np.full(m["n"], k))
Q, C, L, IMG = map(np.concatenate, (Q, C, L, IMG))
# sanity: compare with the earlier mmproj-side token dump where available
try:
    old = {d["stem"]: d for d in S.image_tokens() if d["stem"] in SHOW}
    s0 = next(iter(old))
    a, b = old[s0]["toks"].astype(np.float32), Q[IMG == pick.index(s0)]
    if a.shape == b.shape:
        print("merger recompute vs earlier dump, mean cos:", float(np.mean(np.sum(unit(a) * unit(b), 1))), flush=True)
except Exception as e:  # informational only
    print("no comparison:", e)
naive = Q @ R * scale
obj = L >= 0
tgt = np.array([vi[cat_word[c]] for c in L[obj]])
rng = np.random.default_rng(0)
tgt_shuf = rng.permutation(tgt)
res = dict(n_images=len(pick), n_tokens=int(len(Q)), n_obj_tokens=int(obj.sum()), vocab=len(vocab),
           procrustes_top1=fit_top1)
for name, toks, W in (("qwen", Q, Wq), ("naive", naive, Wl), ("connector", C, Wl)):
    r = rank_of(toks[obj], W, tgt)
    rs = rank_of(toks[obj], W, tgt_shuf)
    nn = np.argmax(unit(toks[obj]) @ unit(W).T, 1)
    top_nn = np.bincount(nn, minlength=len(vocab))
    res[name] = dict(norm=float(np.linalg.norm(toks, axis=1).mean()), word_norm=float(np.linalg.norm(W, axis=1).mean()),
                     hit10=float(np.mean(r < 10)), hit10_shuffled=float(np.mean(rs < 10)),
                     median_rank=float(np.median(r)), median_rank_shuffled=float(np.median(rs)),
                     max_cos_word=float(np.mean(np.max(unit(toks[obj]) @ unit(W).T, 1))),
                     top_nearest=[vocab[i] for i in np.argsort(-top_nn)[:8]])
    print(name, json.dumps(res[name]), flush=True)

# ---------------------------------------------------------------- audio tokens
enc, fe = A.load_whisper()
aconn = A.AudioConnector(d_out=2048).cuda()
aconn.load_state_dict(torch.load(A.OUT / "a1/connector.pt"))
esc = A.Esc()
pool5 = esc.pool({5})
classes = ["dog", "rooster", "rain", "crying baby", "clock alarm", "church bells", "siren", "cow", "thunderstorm", "train"]
arng = random.Random(3)
clips = []
for c in classes:
    idx = [i for i in pool5 if esc.label[i] == c][:4]
    for i in idx:
        x = np.zeros(int(8 * A.SR), np.float32)
        st = arng.uniform(0.5, 2.5)
        clip = esc.wav[i].astype(np.float32); clip = clip / (np.abs(clip).max() or 1) * 0.3
        a0 = int(st * A.SR); x[a0:a0 + len(clip)] += clip[: len(x) - a0]
        clips.append(dict(kind="sound", label=c, x=x, span=(st + esc.on[i], st + esc.off[i])))
for x, txt, i in A.libri_eval(30)[:30]:
    clips.append(dict(kind="speech", label="speech", x=x, span=(0.0, len(x) / A.SR), text=txt))
Wt, At, AL, AK, AT = [], [], [], [], []
with torch.no_grad():
    for k, c in enumerate(clips):
        st = A.encode(enc, fe, [c["x"]])[0]               # [188, 10240]
        out = aconn(st).float().cpu().numpy()
        n = int(np.ceil(len(c["x"]) / A.SR / 0.16))        # tokens covering real audio
        t = (np.arange(n) + 0.5) * 0.16
        inside = (t >= c["span"][0]) & (t <= c["span"][1])
        Wt.append(st[:n].cpu().numpy()[:, ::4]); At.append(out[:n])  # every 4th dim of 10240 keeps PCA light
        AL.append(np.where(inside, c["label"], "background")); AK.append(np.full(n, k)); AT.append(t)
Wt, At, AL, AK = np.concatenate(Wt), np.concatenate(At), np.concatenate(AL), np.concatenate(AK)
snd = np.isin(AL, classes)
cw = {"crying baby": "baby", "clock alarm": "alarm", "church bells": "bells"}
atgt = np.array([vi.get(cw.get(l, l), -1) for l in AL[snd]])
keep = atgt >= 0
r = rank_of(At[snd][keep], Wl, atgt[keep]); rs = rank_of(At[snd][keep], Wl, rng.permutation(atgt[keep]))
nn = np.argmax(unit(At) @ unit(Wl).T, 1)
res["audio"] = dict(n_clips=len(clips), n_tokens=int(len(At)), norm=float(np.linalg.norm(At, axis=1).mean()),
                    hit10=float(np.mean(r < 10)), hit10_shuffled=float(np.mean(rs < 10)),
                    median_rank=float(np.median(r)), median_rank_shuffled=float(np.median(rs)),
                    max_cos_word=float(np.mean(np.max(unit(At) @ unit(Wl).T, 1))),
                    top_nearest=[vocab[i] for i in np.argsort(-np.bincount(nn, minlength=len(vocab)))[:8]])
# speech: is a transcript word among the 10 nearest words of the clip's tokens?
sp_hits, sp_shuf = [], []
spk = [k for k, c in enumerate(clips) if c["kind"] == "speech"]
for j, k in enumerate(spk):
    toks = At[AK == k]
    top = np.argsort(-(unit(toks) @ unit(Wl).T), 1)[:, :10]
    words = {vi[w] for w in A.norm_text(clips[k]["text"]).split() if w in vi}
    other = {vi[w] for w in A.norm_text(clips[spk[(j + 5) % len(spk)]]["text"]).split() if w in vi}
    sp_hits.append(np.mean([len(set(t) & words) > 0 for t in top])); sp_shuf.append(np.mean([len(set(t) & other) > 0 for t in top]))
res["audio"]["speech_token_hits_transcript_word"] = float(np.mean(sp_hits))
res["audio"]["speech_token_hits_other_transcript"] = float(np.mean(sp_shuf))
print("audio", json.dumps(res["audio"]), flush=True)

# ---------------------------------------------------------------- shared 2D map (Llama word space)
# PCA fit with equal weight on: Llama words, connector image tokens, connector audio tokens (all unit-normalised,
# so the map shows direction; lengths are reported separately).
sel_v = rng.choice(len(C), 1500, replace=False); sel_a = rng.choice(len(At), 1000, replace=False)
word_ids = np.arange(min(1500, len(vocab)))
fitset = [unit(Wl[word_ids]), unit(C[sel_v]), unit(At[sel_a])]
mu = np.mean([f.mean(0) for f in fitset], 0)
Z = np.concatenate([f - mu for f in fitset])
_, _, P = np.linalg.svd(Z, full_matrices=False)
proj = lambda v: (unit(v) - mu) @ P[:2].T
# whisper raw tokens: their own 2D PCA (no word space exists for them)
wz = unit(Wt) - unit(Wt).mean(0); _, _, PW = np.linalg.svd(wz[sel_a], full_matrices=False)
# vision "before": Qwen's own space, 2D PCA fitted on Qwen words + Qwen image tokens (equal weight)
qfit = [unit(Wq[word_ids]), unit(Q[sel_v])]
qmu = np.mean([f.mean(0) for f in qfit], 0)
_, _, PQ = np.linalg.svd(np.concatenate([f - qmu for f in qfit]), full_matrices=False)
qproj = lambda v: (unit(v) - qmu) @ PQ[:2].T
r2 = lambda a: np.round(a, 4).tolist()
colors_for = ["person", "car", "dog", "cat", "pizza", "bed", "stop sign", "bus", "giraffe", "train"]
cat_tokens = {}
for nm in colors_for:
    c = names.index(nm)
    ids = np.nonzero(L == c)[0]
    if len(ids):
        ids = rng.choice(ids, min(len(ids), 60), replace=False)
        cat_tokens[nm] = dict(word=cat_word[c], qwen=r2(qproj(Q[ids])), naive=r2(proj(naive[ids])), conn=r2(proj(C[ids])),
                              word_qwen=r2(qproj(Wq[vi[cat_word[c]]][None]))[0], word_llama=r2(proj(Wl[vi[cat_word[c]]][None]))[0])
bg = rng.choice(np.nonzero(L == -1)[0], 300, replace=False)
aud = {}
for c in classes + ["speech"]:
    ids = np.nonzero(AL == c)[0]
    if len(ids):
        ids = rng.choice(ids, min(len(ids), 50), replace=False)
        w = cw.get(c, c)
        aud[c] = dict(word=w, whisper=r2(wz[ids] @ PW[:2].T), conn=r2(proj(At[ids])),
                      word_llama=r2(proj(Wl[vi[w]][None]))[0] if w in vi else None)
res["map"] = dict(words=dict(text=[vocab[i] for i in word_ids[:400]], llama=r2(proj(Wl[word_ids[:400]])),
                             qwen=r2(qproj(Wq[word_ids[:400]]))),
                  vision=cat_tokens, background=dict(qwen=r2(qproj(Q[bg])), naive=r2(proj(naive[bg])), conn=r2(proj(C[bg]))),
                  audio=aud, audio_bg=dict(whisper=r2(wz[np.nonzero(AL == "background")[0][:200]] @ PW[:2].T),
                                           conn=r2(proj(At[np.nonzero(AL == "background")[0][:200]]))))

# ---------------------------------------------------------------- per-token heatmaps for the showcase images
heat, heat_stats = {}, []
for k, s in enumerate(SHOW):
    m = by[s]; sel = IMG == k
    lab = L[sel]
    present = [c for c in np.unique(lab) if c >= 0]
    if not present:
        continue
    c = max(present, key=lambda c: (lab == c).sum())
    w = cat_word[c]
    qs, cs, msk = unit(Q[sel]) @ unit(Wq[vi[w]]), unit(C[sel]) @ unit(Wl[vi[w]]), lab == c
    heat_stats.append(dict(stem=s, word=w, qwen_in=float(qs[msk].mean()), qwen_out=float(qs[~msk].mean()),
                           conn_in=float(cs[msk].mean()), conn_out=float(cs[~msk].mean())))
    heat[s] = dict(gh=m["gh"], gw=m["gw"], cls=names[c], word=w, mask=(lab == c).astype(int).tolist(),
                   qwen=r2(unit(Q[sel]) @ unit(Wq[vi[w]])), conn=r2(unit(C[sel]) @ unit(Wl[vi[w]])),
                   qwen_rank=rank_of(Q[sel], Wq, np.full(sel.sum(), vi[w])).tolist(),
                   conn_rank=rank_of(C[sel], Wl, np.full(sel.sum(), vi[w])).tolist())
res["heat"] = heat
# all 80 images: is an object's own tokens' similarity to its word higher than other tokens'?
win = {"qwen": [], "conn": []}
for k in range(len(pick)):
    sel = IMG == k; lab = L[sel]
    for c in [c for c in np.unique(lab) if c >= 0 and (lab == c).sum() >= 3]:
        w = vi[cat_word[c]]
        for nm, toks, W in (("qwen", Q, Wq), ("conn", C, Wl)):
            sim = unit(toks[sel]) @ unit(W[w])
            pos, neg = sim[lab == c], sim[lab != c]
            win[nm].append(float((pos[:, None] > neg[None, :]).mean()))  # AUC: object tokens vs the rest
res["word_auc"] = {k: float(np.mean(v)) for k, v in win.items()}
res["word_auc_n"] = len(win["qwen"])
res["heat_stats"] = heat_stats
print("word-similarity AUC (object tokens vs rest):", res["word_auc"], res["word_auc_n"], flush=True)
json.dump(res, open(OUT, "w"))
print("wrote", OUT)
