"""Training-free "layer transplant" test (user's idea): copy the first k LLM layers of a real VLM
(Qwen3-VL-2B, frozen, including its DeepStack injections), then map the image tokens into Llama-3.2-1B's
input space without any training, and see whether the unmodified Llama recognises what is in the image.

Bridges from Qwen hidden state h_k (image tokens after k Qwen layers) to Llama input vectors:
  A  rotation: orthogonal Procrustes fitted on the shared word-embedding vocabularies (Qwen3-VL -> Llama),
     each vector rescaled to Llama's mean word-embedding length
  B  decode + re-embed: logit-lens decode of h_k to Qwen's top-5 tokens, mapped by identical token
     string to Llama tokens, probability-weighted average of their Llama embeddings
References: no image at all; Qwen3-VL-2B answering itself; our trained connector (run_full).

usage: bridge_test.py qwen    # phase 1 (GPU: Qwen3-VL only) -> analysis/bridge/*.npz + qwen answers
       bridge_test.py llama   # phase 2 (GPU: Llama only)    -> analysis/bridge/results.json
"""
import json
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

import eval_extra as E
import train_connector as T

OUT = T.ROOT / "analysis/bridge"
OUT.mkdir(parents=True, exist_ok=True)
MD = T.ROOT / "dl/qwen3vl-2b"
KS = [0, 1, 2, 4, 12, 20, 24, 28]
N = 100
Q = "What objects are in this image? Answer with a short list."
SYN = {"person": ["man", "woman", "people", "boy", "girl", "child", "kid", "player", "men", "women", "skier", "surfer", "person", "persons", "lady", "guy"],
       "tv": ["television", "tv", "screen", "monitor"], "couch": ["sofa", "couch"], "dining table": ["table"], "cell phone": ["phone", "cellphone"],
       "potted plant": ["plant", "plants"], "sports ball": ["ball"], "motorcycle": ["motorcycle", "motorbike"], "airplane": ["airplane", "plane", "aircraft", "jet"],
       "hair drier": ["dryer", "drier"], "laptop": ["laptop", "computer"], "bicycle": ["bicycle", "bike"], "handbag": ["handbag", "bag", "purse"],
       "skis": ["ski", "skis"], "donut": ["donut", "doughnut"], "wine glass": ["glass", "wine"], "teddy bear": ["teddy"], "hot dog": ["hot dog", "hotdog"],
       "fire hydrant": ["hydrant"], "stop sign": ["stop sign"], "traffic light": ["traffic light"], "parking meter": ["parking meter"],
       "baseball bat": ["bat"], "baseball glove": ["glove"], "tennis racket": ["racket", "racquet"], "remote": ["remote"], "mouse": ["mouse"]}


def eval_set():
    meta = json.load(open(T.DATA / "meta.json"))
    by = {m["stem"]: m for m in meta["metas"]}
    ev = [m for m in [by[s] for s in meta["split"]["eval"]][:200] if E.gt_of(m)]
    return ev[:N]


def mentions(text, names):
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    out = set()
    for n in names:
        for w in SYN.get(n, [n]):
            if re.search(rf"\b{re.escape(w)}(s|es)?\b", t):
                out.add(n); break
    return out


def score(text, m, names):
    gt = {o["name"] for o in m["objs"] if o["area"] >= 0.01 * m["W"] * m["H"]}
    said = mentions(text, names)
    return dict(recall=len(said & gt) / max(1, len(gt)), precision=len(said & gt) / len(said) if said else None,
                any_coco=bool(said), n_gt=len(gt))


# ---------------------------------------------------------------- phase 1: Qwen3-VL
def phase_qwen():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MD)
    proc.image_processor.size = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}
    model = Qwen3VLForConditionalGeneration.from_pretrained(MD, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    L = len(lm.layers)
    cap = {}
    for i, layer in enumerate(lm.layers):
        layer.register_forward_pre_hook(lambda m, args, i=i: cap.__setitem__(i, args[0].detach()))
    lm.norm.register_forward_pre_hook(lambda m, args: cap.__setitem__(L, args[0].detach()))
    img_tok = model.config.image_token_id
    answers = {}
    with torch.no_grad():
        for m in eval_set():
            img = Image.open(T.ROOT / "coco/val2017" / f"{m['stem']}.jpg").convert("RGB")
            s = 448 / max(img.size)
            img = img.resize(tuple(max(32, round(v * s / 32) * 32) for v in img.size), Image.BICUBIC)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            cap.clear()
            model(**inp)
            pos = (inp["input_ids"][0] == img_tok).nonzero().squeeze(1)
            st = {k: cap[k][0, pos].float() for k in KS}
            dec = {}
            for k in KS:
                logits = model.lm_head(lm.norm(st[k].to(torch.bfloat16))).float()
                p, ids = F.softmax(logits, -1).topk(5, -1)
                dec[k] = (ids.cpu().numpy(), p.cpu().numpy())
            np.savez(OUT / f"{m['stem']}.npz", **{f"h{k}": st[k].half().cpu().numpy() for k in KS},
                     **{f"ids{k}": dec[k][0] for k in KS}, **{f"p{k}": dec[k][1] for k in KS})
            o = model.generate(**inp, max_new_tokens=48, do_sample=False)
            answers[m["stem"]] = proc.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(m["stem"], answers[m["stem"]][:80], flush=True)
    json.dump(answers, open(OUT / "qwen_answers.json", "w"), indent=1)


# ---------------------------------------------------------------- phase 2: Llama
def phase_llama():
    import spaces as S
    tok, llm = T.load_llm("cuda")
    Emb = llm.get_input_embeddings().weight
    names, _, _ = S.coco_categories()
    # Qwen3-VL vocab -> Llama vocab by identical token string (both byte-level BPE)
    qv = json.load(open(MD / "tokenizer.json"))["model"]["vocab"]
    lv = json.load(open(T.ROOT / "dl/llama32-1b-tokenizer.json"))["model"]["vocab"]
    q2l = {qi: lv[s] for s, qi in qv.items() if s in lv}
    # Procrustes on shared lower-case word tokens
    pat = re.compile(r"^Ġ[a-z]{3,}$")
    shared = [s for s in qv if pat.match(s) and s in lv]
    with safe_open(MD / "model.safetensors", framework="pt") as f:
        key = next(k for k in f.keys() if k.endswith("language_model.embed_tokens.weight"))
        QE = f.get_tensor(key)                 # bf16, 0.6 GB: keep only the shared-word rows
        A = QE[[qv[s] for s in shared]].float()
        del QE
    B = Emb.float().cpu()[[lv[s] for s in shared]]
    U, _, Vt = torch.linalg.svd(A.T @ B)
    R = (U @ Vt).cuda()
    hold = torch.arange(len(shared)) % 10 == 0
    top1 = float(((F.normalize(A[hold] @ R.cpu(), dim=-1) @ F.normalize(B, dim=-1).T).argmax(1) == hold.nonzero().squeeze(1)).float().mean())
    lnorm = Emb.float().norm(dim=-1).mean()
    print(f"{len(shared)} shared words, rotation word top-1 {top1:.4f} (fit set sample), {len(q2l)} mapped tokens", flush=True)

    P = T.Batcher(tok, None, None, None)
    eot = P.eot

    def gen(img_vecs, q=Q):
        msgs = [{"role": "user", "content": ("<__media__>" if img_vecs is not None else "") + q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if img_vecs is None:
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            emb = Emb[ids[0]][None]
        else:
            pre, post = text.split("<__media__>")
            e = lambda s: Emb[tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0].cuda()]
            emb = torch.cat([e(pre), img_vecs.to(Emb.dtype), e(post)])[None]
        o = llm.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device="cuda"),
                         max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=[eot, tok.eos_token_id])
        return tok.decode(o[0], skip_special_tokens=True).strip()

    ev = eval_set()
    qa = json.load(open(OUT / "qwen_answers.json"))
    X = np.load(T.DATA / "x.npy", mmap_mode="r")
    conn = T.Connector(2048).cuda()
    conn.load_state_dict(torch.load(T.DATA / "run_full/connector.pt"))
    rows = {}
    with torch.no_grad():
        for j, m in enumerate(ev):
            z = np.load(OUT / f"{m['stem']}.npz")
            r = {"no_image": gen(None), "qwen3vl_itself": qa[m["stem"]]}
            x = torch.from_numpy(np.asarray(X[m["off"]:m["off"] + m["n"]], dtype=np.float32)).cuda()
            pre, post = P.prompt_parts(Q)   # trained connector: its own image markers, same question
            embc, attc, _ = T.embed_batch(llm, conn, [(pre, x, post, [])], "cuda")
            o = llm.generate(inputs_embeds=embc, attention_mask=attc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
            r["trained_connector"] = tok.decode(o[0], skip_special_tokens=True).strip()
            for k in KS:
                h = torch.from_numpy(z[f"h{k}"]).float().cuda()
                v = h @ R
                r[f"A_rotate_k{k}"] = gen(v / v.norm(dim=-1, keepdim=True) * lnorm)
                ids, p = z[f"ids{k}"], z[f"p{k}"]
                vecs = []
                for ii, pp in zip(ids, p):
                    lid = [(q2l[int(a)], float(b)) for a, b in zip(ii, pp) if int(a) in q2l]
                    if lid:
                        w = torch.tensor([b for _, b in lid], device="cuda")
                        vecs.append((Emb[[a for a, _ in lid]].float() * (w / w.sum())[:, None]).sum(0))
                r[f"B_decode_k{k}"] = gen(torch.stack(vecs)) if vecs else ""
            rows[m["stem"]] = r
            if j % 10 == 0:
                print(j, {k: v[:60] for k, v in r.items() if k in ("no_image", "A_rotate_k2", "B_decode_k24", "trained_connector")}, flush=True)
    conds = list(next(iter(rows.values())).keys())
    summary = {}
    for c in conds:
        sc = [score(rows[m["stem"]][c], m, names) for m in ev]
        prec = [s["precision"] for s in sc if s["precision"] is not None]
        summary[c] = dict(recall=float(np.mean([s["recall"] for s in sc])), precision=float(np.mean(prec)) if prec else None,
                          names_any_coco_object=float(np.mean([s["any_coco"] for s in sc])))
        print(f"{c:22s} recall {summary[c]['recall']:.3f}  precision {summary[c]['precision'] if summary[c]['precision'] is None else round(summary[c]['precision'], 3)}  any {summary[c]['names_any_coco_object']:.2f}", flush=True)
    json.dump(dict(summary=summary, rows=rows, n=len(ev), question=Q, ks=KS, rotation_top1=top1), open(OUT / "results.json", "w"), indent=1)


if __name__ == "__main__":
    {"qwen": phase_qwen, "llama": phase_llama}[sys.argv[1]]()
