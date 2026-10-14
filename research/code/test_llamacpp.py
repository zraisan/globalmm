"""Grounding test through llama-server (text-only Llama GGUF + our mmproj).

Resizes each image the same way as training (longest side 448, multiples of 32), asks the server,
parses [x1,y1,x2,y2] boxes (0..999) and draws them (red) next to COCO ground truth (green).

usage: test_llamacpp.py PORT OUT_DIR STEM...
"""
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
port, out, stems = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
url = f"http://127.0.0.1:{port}"
meta = {m["stem"]: m for m in json.load(open(ROOT / "analysis/grounding/meta.json"))["metas"]}

for _ in range(180):
    try:
        if requests.get(url + "/health", timeout=2).status_code == 200:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    sys.exit("server not healthy")


def fit32(img, side=448):
    s = side / max(img.size)
    return img.resize(tuple(max(32, round(v * s / 32) * 32) for v in img.size), Image.BICUBIC)


def ask(img, q, n=160):
    buf = io.BytesIO(); img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    r = requests.post(url + "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": q}]}],
        "max_tokens": n, "temperature": 0, "seed": 0}, timeout=600).json()
    return r["choices"][0]["message"]["content"].strip(), r.get("usage", {}).get("prompt_tokens")


def boxes(t):
    return [tuple(int(v) for v in b) for b in re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", t)]


results = {}
for s in stems:
    m = meta[s]
    src = Image.open(ROOT / "coco/val2017" / f"{s}.jpg").convert("RGB")
    img = fit32(src)
    objs = sorted([o for o in m["objs"] if o["area"] >= 0.004 * m["W"] * m["H"]], key=lambda o: -o["area"])
    main = objs[0]["name"] if objs else "object"
    r = {}
    r["list"], ntok = ask(img, "List the objects in this image with their bounding boxes.")
    r["where_q"] = main
    r["where"], _ = ask(img, f"Where is the {main}? Answer with bounding boxes.", 60)
    r["desc"], _ = ask(img, "Describe this image in one sentence.", 60)
    r["prompt_tokens"] = ntok
    results[s] = r
    # draw: left = list answer, right = where answer; green = COCO truth
    W, H = src.size
    sc = lambda b: (b[0] / 999 * W, b[1] / 999 * H, b[2] / 999 * W, b[3] / 999 * H)
    panels = []
    for key in ("list", "where"):
        p = src.copy(); d = ImageDraw.Draw(p)
        for o in objs if key == "list" else [o for o in objs if o["name"] == main]:
            x, y, w, h = o["box"]; d.rectangle([x, y, x + w, y + h], outline=(40, 200, 80), width=3)
        for line in r[key].splitlines() if key == "list" else [r[key]]:
            for b in boxes(line):
                b = (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
                d.rectangle(sc(b), outline=(230, 40, 40), width=3)
                lab = re.sub(r"\s*\[.*", "", line).strip()[:20] if key == "list" else main
                d.text((sc(b)[0] + 4, sc(b)[1] + 2), lab, fill=(230, 40, 40))
        panels.append(p)
    canvas = Image.new("RGB", (W * 2 + 10, H), "white")
    canvas.paste(panels[0], (0, 0)); canvas.paste(panels[1], (W + 10, 0))
    canvas.save(out / f"{s}_boxes.jpg", quality=88)
    print(f"### {s} (prompt tokens {ntok})\nLIST: {r['list']!r}\nWHERE {main}: {r['where']}\nDESC: {r['desc']}\n", flush=True)
json.dump(results, open(out / "results.json", "w"), indent=1)
