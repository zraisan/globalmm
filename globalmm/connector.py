"""
`--method connector`: train a new projector for an existing mmproj and a target LLM.

The mmproj's vision tower and the target LLM stay frozen. Only a projector with the same shape as
the mmproj's own projector is trained, with next-token loss on answers that contain bounding boxes
and captions (see coco.py). Gradients flow through the frozen LLM into the projector. The result is
written as a new mmproj: the original tower plus the trained projector, which llama.cpp loads
without code changes.

Training is resumable: a checkpoint is written to the work directory every 200 steps.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .coco import Coco, ensure_annotations, ensure_images, make_example
from .towers import Projector, export_mmproj, open_tower

MARKER = "<__media__>"


class Prompts:
    """Token ids around the image and after it, built from the LLM's own chat template."""

    def __init__(self, tok, wrap: tuple[str, str]):
        self.tok, self.wrap = tok, wrap

    def parts(self, question: str, answer: str | None = None):
        user = [{"role": "user", "content": MARKER + question}]
        prompt = self.tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        pre, post = prompt.split(MARKER)
        pre_ids = self.tok(pre + self.wrap[0], add_special_tokens=False).input_ids
        post_ids = self.tok(self.wrap[1] + post, add_special_tokens=False).input_ids
        if answer is None:
            return pre_ids, post_ids, []
        full = self.tok.apply_chat_template(user + [{"role": "assistant", "content": answer}], tokenize=False)
        tail = full[len(prompt):] if full.startswith(prompt) else answer + (self.tok.eos_token or "")
        return pre_ids, post_ids, self.tok(tail, add_special_tokens=False).input_ids


def _embed(llm, proj, items, device):
    """items: (pre_ids, feats, post_ids, ans_ids) -> inputs_embeds, attention_mask, labels."""
    E = llm.get_input_embeddings()
    seqs, labs = [], []
    for pre, x, post, ans in items:
        img = proj(x.to(device)).to(E.weight.dtype)
        tok = lambda ids: E(torch.tensor(ids, device=device, dtype=torch.long))
        emb = torch.cat([tok(pre), img, tok(post + ans)])
        lab = torch.full((len(emb),), -100, dtype=torch.long, device=device)
        if ans:
            lab[-len(ans):] = torch.tensor(ans, device=device)
        seqs.append(emb)
        labs.append(lab)
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, seqs[0].shape[1], dtype=seqs[0].dtype, device=device)
    att = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long, device=device)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        out[i, :len(s)] = s
        att[i, :len(s)] = 1
        lab[i, :len(s)] = l
    return out, att, lab


def _load_llm(name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    return tok, llm


def train_connector(mmproj: Path, llm_name: str, out: Path, coco_dir: Path, coco_images: Path | None,
                    work_dir: Path, steps: int | None, batch: int = 2, accum: int = 4, lr: float = 3e-4,
                    warmup: int = 100, max_images: int | None = None, seed: int = 0) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("--method connector needs a CUDA GPU (about 8 GB for a 1B LLM)")
    device = "cuda"
    work_dir.mkdir(parents=True, exist_ok=True)

    # data
    meta = ensure_annotations(coco_dir)
    images = coco_images or coco_dir / "train448"
    ds = Coco(meta, images)
    stems = ds.stems if max_images is None else [ds.stems[i] for i in np.random.default_rng(seed).permutation(len(ds.stems))[:max_images]]
    if coco_images is None:
        ensure_images(stems, images)
    keep = set(stems)
    idx = [i for i in ds.available() if ds.stems[i] in keep]
    per_step = batch * accum
    steps = steps or math.ceil(len(idx) / per_step)  # default: one pass over the images
    print(f"{len(idx)} training images, {steps} steps of {per_step} examples")

    # models
    tower, reader = open_tower(mmproj, device)
    tok, llm = _load_llm(llm_name, device)
    d_llm = llm.get_input_embeddings().weight.shape[1]
    proj = Projector(tower.spec, d_llm)
    opt = None
    ck = work_dir / "ckpt.pt"
    step = 0
    if ck.exists():
        s = torch.load(ck, map_location="cpu")
        proj.load_state_dict(s["proj"])
        step = s["step"]
        proj.to(device)
        opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
        opt.load_state_dict(s["opt"])
        print(f"resumed at step {step}")
    else:
        # first layer starts from the mmproj's own projector; second is random, scaled so outputs start
        # at the size of the LLM's word embeddings (measured on real features)
        with torch.no_grad():
            proj.fc1.weight.copy_(tower.init_fc1[0])
            proj.fc1.bias.copy_(tower.init_fc1[1])
            torch.nn.init.normal_(proj.fc2.weight, std=0.02)
            torch.nn.init.zeros_(proj.fc2.bias)
            proj.to(device)
            sample = torch.cat(tower.features([tower.preprocess(tower.fit(ds.image(i))) for i in idx[:4]]))
            emb = llm.get_input_embeddings().weight.float().norm(dim=-1).mean()
            proj.fc2.weight.mul_(emb / proj(sample).norm(dim=-1).mean())
        opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    json.dump(dict(mmproj=str(mmproj), llm=llm_name, steps=steps, batch=batch, accum=accum, lr=lr,
                   images=len(idx)), open(work_dir / "config.json", "w"), indent=1)

    P = Prompts(tok, tower.image_wrap)
    order = [idx[j] for j in np.random.default_rng(seed + 1).permutation(len(idx))]
    ptr = step * per_step
    rng = random.Random(seed + 1000 + step)
    sched = lambda s: min(1.0, s / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    t0, start, run = time.time(), step, []
    while step < steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = lr * sched(step)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            pix, qa = [], []
            while len(pix) < batch:
                i = order[ptr % len(order)]
                ptr += 1
                try:
                    img = tower.fit(ds.image(i))
                except OSError:
                    continue
                pix.append(tower.preprocess(img))
                qa.append(make_example(ds.meta(i), rng))
            feats = tower.features(pix)
            items = []
            for (q, a), x in zip(qa, feats):
                pre, post, ans = P.parts(q, a)
                items.append((pre, x, post, ans))
            emb, att, lab = _embed(llm, proj, items, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hid = llm.model(inputs_embeds=emb, attention_mask=att).last_hidden_state
                tgt = lab[:, 1:]
                sel = tgt != -100   # vocabulary projection only where there is a target
                loss = F.cross_entropy(llm.lm_head(hid[:, :-1][sel]).float(), tgt[sel]) / accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
        run.append(total)
        if step % 50 == 0 or step == steps:
            rate = (time.time() - t0) / (step - start)
            print(f"step {step}/{steps}  loss {np.mean(run[-50:]):.3f}  {rate:.2f} s/step  "
                  f"eta {(steps - step) * rate / 3600:.1f} h  vram {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)
        if step % 200 == 0 or step == steps:
            tmp = work_dir / "ckpt.tmp"
            torch.save(dict(proj=proj.state_dict(), opt=opt.state_dict(), step=step), tmp)
            tmp.rename(ck)

    export_mmproj(reader, tower, proj, out)
    print(f"wrote {out} (projection width {d_llm}); load it with the {llm_name} GGUF in llama.cpp")
