"""Figures for research/README.md (light theme, recessive grid, direct labels)."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent / "research/figures"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"
C1, C2, C3 = "#2f6fdf", "#e0782f", "#7a7f87"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def curve(log, every_re=r"^step (\d+)/\d+\s+loss ([\d.]+)"):
    xs, ys = [], []
    for ln in open(log):
        m = re.match(every_re, ln)
        if m:
            xs.append(int(m[1])); ys.append(float(m[2]))
    return xs, ys


fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
x1, y1 = curve(ROOT / "analysis/grounding/run1.log")
x2, y2 = curve(ROOT / "analysis/grounding/run_full.log")
ax[0].plot([x * 8 / 1000 for x in x1], y1, color=C3, lw=2, label="run1 (4.5k val2017 images)")
ax[0].plot([(2500 + x) * 8 / 1000 for x in x2], y2, color=C1, lw=2, label="full (118k train2017, init from run1)")
ax[0].set(title="Image grounding connector", xlabel="training examples (thousands)", ylabel="answer-token loss")
ax[0].legend(frameon=False)
xa, ya = curve(ROOT / "analysis/audio/a1.log")
ax[1].plot([x * 8 / 1000 for x in xa], ya, color=C2, lw=2, label="audio connector (speech + sounds)")
ax[1].set(title="Audio connector", xlabel="training examples (thousands)", ylabel="answer-token loss")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "loss.png", dpi=150, facecolor="white")
print("loss.png")


# ---------------------------------------------------------------- Acc@0.5 by object size (in-distribution)
G = ROOT / "analysis/grounding"
q = G / "qwen3vl_ref.json"
if q.exists():
    rows = [("run1", json.load(open(G / "run1/eval_extra.json"))["image"]["summary"], C3),
            ("full", json.load(open(G / "run_full/eval_extra.json"))["image"]["summary"], C1),
            ("Qwen3-VL-2B (reference)", json.load(open(q))["summary"], C2)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sizes = ["small", "medium", "large"]
    w = 0.26
    for j, (name, s, c) in enumerate(rows):
        vals = [100 * (s.get(f"where_acc50_{k}") or 0) for k in sizes]
        bars = ax.bar([i + (j - 1) * w for i in range(3)], vals, w - 0.03, color=c, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), ["small (<1%, n=9)", "medium (1–10%, n=85)", "large (>10%, n=98)"])
    ax.set(ylabel="Acc@0.5 (%)", ylim=(0, 105), title='"Where is the X?" by object size (held-out COCO)')
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "acc_by_size.png", dpi=150, facecolor="white")
    print("acc_by_size.png")

# ---------------------------------------------------------------- out-of-distribution examples
ood_p, qref = G / "run_full/eval_ood.json", G / "qwen3vl_ref_lvis.json"
if ood_p.exists() and qref.exists():
    from PIL import Image, ImageDraw
    ours = {r["stem"]: r for r in json.load(open(ood_p))["image"]["rows"]}
    qw = {r["stem"]: r for r in json.load(open(qref))["rows"]}
    gt = {r["stem"]: r for r in json.load(open(G / "lvis_ood.json"))}
    pick = [s for s in gt if gt[s]["size"] >= 0.04][:: max(1, len([s for s in gt if gt[s]["size"] >= 0.04]) // 6)][:6]
    tiles = []
    for s in pick:
        im = Image.open(ROOT / "coco/val2017" / f"{s}.jpg").convert("RGB")
        im.thumbnail((420, 420)); W, H = im.size
        d = ImageDraw.Draw(im)
        sc = lambda b: (b[0] / 999 * W, b[1] / 999 * H, b[2] / 999 * W, b[3] / 999 * H)
        for b in gt[s]["boxes"]:
            d.rectangle(sc(b), outline=(40, 200, 80), width=3)
        pb = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", ours[s]["answer"])
        if pb:
            d.rectangle(sc([int(v) for v in pb[0]]), outline=(230, 40, 40), width=3)
        qb = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", qw[s]["answer"])
        if qb:
            d.rectangle(sc([int(v) * 999 / 1000 for v in qb[0]]), outline=(40, 110, 230), width=2)
        d.rectangle((0, 0, W, 18), fill=(255, 255, 255))
        d.text((4, 3), f'"{gt[s]["name"]}"   ours IoU {ours[s]["iou"]:.2f} | Qwen3-VL {qw[s]["where_iou"]:.2f}',
               fill=(20, 20, 20))
        tiles.append(im)
    cw, ch = 420, max(t.size[1] for t in tiles)
    canvas = Image.new("RGB", (cw * 3 + 20, ch * 2 + 10), "white")
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % 3) * (cw + 10), (i // 3) * (ch + 10)))
    canvas.save(FIG / "ood_examples.jpg", quality=88)
    print("ood_examples.jpg")

# ---------------------------------------------------------------- audio timelines (held-out ESC-50 fold-5 mixtures)
ae = ROOT / "analysis/audio/a1/eval.json"
if ae.exists():
    rows = [r for r in json.load(open(ae))["raw"]["when_audio"] if len(r["events"]) >= 2][:6]
    fig, axs = plt.subplots(len(rows), 1, figsize=(9, 0.95 * len(rows) + 0.6), sharex=True)
    for ax, r in zip(axs, rows):
        labs = [e[0] for e in r["events"]]
        pred = {}
        for ln in r["list"].splitlines():
            m = re.match(r"\s*(.*?)\s*[\[(](\d+(?:\.\d+)?)s?,\s*(\d+(?:\.\d+)?)s?[\])]", ln)
            if m:
                pred[m[1]] = (float(m[2]), float(m[3]))
        names = labs + [n for n in pred if n not in labs]
        for i, n in enumerate(names):
            for e in r["events"]:
                if e[0] == n:
                    ax.broken_barh([(e[1], e[2] - e[1])], (i + 0.52, 0.36), color="#2e9e57")
            if n in pred:
                s, e = pred[n]
                ax.broken_barh([(s, e - s)], (i + 0.12, 0.36), color="#d6453d")
        ax.set_yticks([i + 0.5 for i in range(len(names))], names, fontsize=8)
        ax.set_ylim(0, len(names)); ax.set_xlim(0, 25)
        ax.axvline(r["T"], color=MUTED, lw=1, ls=":")
        ax.grid(axis="y", visible=False)
    axs[-1].set_xlabel("seconds   (green = ground truth, red = model's list answer, dotted = end of clip)")
    axs[0].set_title('"List the sounds in this audio with their timestamps." (held-out ESC-50 mixtures)')
    fig.tight_layout(); fig.savefig(FIG / "audio_timelines.png", dpi=150, facecolor="white")
    print("audio_timelines.png")
