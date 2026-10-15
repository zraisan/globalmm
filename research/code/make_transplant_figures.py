"""Figures and numbers for the connector + donor section of research/README.md (Section 8).

  transplant_sweep.png     COCO / LVIS Acc@0.5 vs donor layers k (3k-step runs) with the matched baseline
  equal_footing.png        full-data runs: loss curves + held-out scores, connector + donor (k=4) vs connector (Qwen3.8)
  transplant_examples.jpg  "where is X?" answers of both full-data runs on evenly spaced held-out images
Also prints Acc@0.5 by object size and a paired (McNemar) comparison of the two full-data runs.
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_extra as E

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent / "research/figures"
G = ROOT / "analysis/grounding"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"   # same style as make_figures.py
C1, C2 = "#2f6fdf", "#e0782f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def curve(log):
    xs, ys = [], []
    for ln in open(log):
        m = re.match(r"^step (\d+)/\d+\s+loss ([\d.]+)", ln)
        if m:
            xs.append(int(m[1])); ys.append(float(m[2]))
    return xs, ys


def load(tag, f):
    return json.load(open(G / tag / f))


sweep = {k: load(f"transplant_k{k}", "eval.json") for k in (0, 4, 12)}
matched = load("matched_qwen38", "eval_hybrid.json")
eq_t, eq_b = load("eq_transplant_k4", "eval.json"), load("eq_qwen38", "eval_hybrid.json")

# ---------------------------------------------------------------- k sweep (3k-step runs)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, part, title in ((ax[0], "coco", "COCO (80 trained categories)"), (ax[1], "lvis", "LVIS (non-COCO objects)")):
    ks = sorted(sweep)
    vals = [100 * sweep[k][part]["where_acc50"] for k in ks]
    a.plot(ks, vals, color=C1, lw=2, marker="o", ms=8, label="connector + donor (Qwen3-VL-2B tower + k layers)")
    m = 100 * matched[part]["where_acc50"]
    for k, v in zip(ks, vals):  # value labels above the marker, or below it when under the baseline line
        a.text(k, v + 2 if v > m else v - 2, f"{v:.1f}", ha="center", va="bottom" if v > m else "top", fontsize=9, color=INK)
    a.axhline(m, color=C2, lw=2, ls="--", label="connector (Qwen3.8 tower), same budget")
    a.text(8, m - 0.8, f"{m:.1f}", ha="center", va="top", fontsize=9, color=INK)
    a.set(title=title, xlabel="donor LLM layers copied (k)", ylabel='"where" Acc@0.5 (%)', xticks=[0, 4, 12],
          ylim=(0, max(vals) + 15), xlim=(-1, 14))
ax[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "transplant_sweep.png", dpi=150, facecolor="white")
print("transplant_sweep.png")

# ---------------------------------------------------------------- equal footing
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
for log, c, name in (("train_transplant.log", C1, "connector + donor (k=4)"), ("train_baseline.log", C2, "connector (Qwen3.8)")):
    xs, ys = curve(ROOT / "analysis/equal" / log)
    ax[0].plot([x * 8 / 1000 for x in xs], ys, color=c, lw=2, label=name)
ax[0].set(title="Training loss (same data and schedule)", xlabel="training examples (thousands)", ylabel="answer-token loss")
ax[0].legend(frameon=False)
metrics = [("COCO\nAcc@0.5", "coco", "where_acc50"), ("COCO list\nbox recall", "coco", "box_r"),
           ("LVIS\nAcc@0.5", "lvis", "where_acc50"), ("LVIS large\nAcc@0.5", "lvis", "where_acc50_large")]
w = 0.38
for j, (e, c, name) in enumerate(((eq_t, C1, "connector + donor (k=4)"), (eq_b, C2, "connector (Qwen3.8)"))):
    vals = [100 * e[p][k] for _, p, k in metrics]
    bars = ax[1].bar([i + (j - 0.5) * w for i in range(len(metrics))], vals, w - 0.04, color=c, label=name)
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
ax[1].set_xticks(range(len(metrics)), [m[0] for m in metrics])
ax[1].set(ylabel="%", ylim=(0, 100), title="Held-out scores")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "equal_footing.png", dpi=150, facecolor="white")
print("equal_footing.png")

# ---------------------------------------------------------------- by size + paired test
meta = {m["stem"]: m for m in json.load(open(G / "meta.json"))["metas"]}


def size_of(r):
    return max(a for _, a in E.gt_of(meta[r["stem"]])[r["name"]])


for name, e in (("transplant", eq_t), ("qwen38", eq_b)):
    out = []
    for lab, lo, hi in (("small", 0, .01), ("medium", .01, .1), ("large", .1, 2)):
        rs = [r for r in e["coco_rows"] if lo <= size_of(r) < hi]
        out.append(f"{lab} {100 * sum(r['where_iou'] >= .5 for r in rs) / len(rs):.0f}% (n={len(rs)})")
    print(name, "COCO by size:", ", ".join(out))
for part, key in (("coco_rows", "where_iou"), ("lvis_rows", "iou")):
    a = {(r["stem"], r["name"]): r[key] >= .5 for r in eq_t[part]}
    b = {(r["stem"], r["name"]): r[key] >= .5 for r in eq_b[part]}
    only_t = sum(a[q] and not b[q] for q in a)
    only_b = sum(b[q] and not a[q] for q in a)
    n = only_t + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_t, only_b) + 1)) / 2 ** n)
    print(f"{part}: only transplant right {only_t}, only Qwen3.8 right {only_b}, McNemar exact p = {p:.4f}")

# ---------------------------------------------------------------- qualitative
from PIL import Image, ImageDraw

bt = {r["stem"]: r for r in eq_b["coco_rows"]}
# questions the two runs disagree on (Acc@0.5): 4 evenly spaced where only the transplant is right,
# 2 where only the Qwen3.8 connector is right (31 vs 8 such questions in total)
win = [r for r in eq_t["coco_rows"] if r["where_iou"] >= .5 > bt[r["stem"]]["where_iou"]]
lose = [r for r in eq_t["coco_rows"] if bt[r["stem"]]["where_iou"] >= .5 > r["where_iou"]]
rows = win[:: len(win) // 4][:4] + lose[:: len(lose) // 2][:2]
tiles = []
for r in rows:
    im = Image.open(ROOT / "coco/val2017" / f"{r['stem']}.jpg").convert("RGB")
    im.thumbnail((420, 420)); W, H = im.size
    d = ImageDraw.Draw(im)
    sc = lambda b: (b[0] / 999 * W, b[1] / 999 * H, b[2] / 999 * W, b[3] / 999 * H)
    for g, _ in E.gt_of(meta[r["stem"]])[r["name"]]:
        d.rectangle(sc(g), outline=(40, 200, 80), width=3)
    for ans, col, wd in ((bt[r["stem"]]["where"], (230, 130, 40), 2), (r["where"], (40, 110, 230), 3)):
        pb = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", ans)
        if pb:
            d.rectangle(sc([int(v) for v in pb[0]]), outline=col, width=wd)
    d.rectangle((0, 0, W, 18), fill=(255, 255, 255))
    d.text((4, 3), f'"{r["name"]}"   connector + donor IoU {r["where_iou"]:.2f} | connector {bt[r["stem"]]["where_iou"]:.2f}',
           fill=(20, 20, 20))
    tiles.append(im)
cw, ch = 420, max(t.size[1] for t in tiles)
canvas = Image.new("RGB", (cw * 3 + 20, ch * 2 + 10), "white")
for i, t in enumerate(tiles):
    canvas.paste(t, ((i % 3) * (cw + 10), (i // 3) * (ch + 10)))
canvas.save(FIG / "transplant_examples.jpg", quality=88)
print("transplant_examples.jpg")
