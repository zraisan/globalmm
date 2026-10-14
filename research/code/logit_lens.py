"""Where inside a Qwen VLM do image tokens become words?  Logit lens on Qwen3-VL-2B.

For every image token we read the residual stream at the input of each decoder layer
(0 = merger output as fed to the LLM, ..., 28 = after the last layer), push it through the
final RMSNorm + lm_head, and ask whether the COCO object under that token is named.

usage: logit_lens.py IMAGE_LIST [max_images] [--no-deepstack] [--tag NAME]
Writes analysis/results/lens_<tag>.json and analysis/lens_maps_<tag>.npz
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import spaces as S

p = argparse.ArgumentParser()
p.add_argument("image_list")
p.add_argument("max_images", type=int, nargs="?", default=300)
p.add_argument("--no-deepstack", action="store_true")
p.add_argument("--tag", default="full")
a = p.parse_args()

MD = S.DL / "qwen3vl-2b"
TESTS = {"000000039769": "cat", "000000000139": "tv", "000000000785": "person",
         "000000000632": "bed", "000000001000": "tennis racket", "000000000724": "stop sign"}
SYN = {"person": ["man", "woman", "people", "boy", "girl", "child", "kid", "player", "men", "women", "skier", "surfer"],
       "tv": ["television", "screen", "monitor"], "couch": ["sofa"], "dining table": ["table"], "cell phone": ["phone"],
       "potted plant": ["plant", "plants"], "sports ball": ["ball"], "motorcycle": ["motorbike", "bike"],
       "airplane": ["plane", "aircraft", "jet"], "hair drier": ["dryer"], "laptop": ["computer"], "bicycle": ["bike"],
       "handbag": ["bag", "purse"], "skis": ["ski"], "snowboard": ["board"], "surfboard": ["board", "surf"],
       "donut": ["doughnut"], "toothbrush": ["brush"], "frisbee": ["disc"]}

ZH = {"person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车", "airplane": "飞机", "bus": "公交车", "train": "火车",
      "truck": "卡车", "boat": "船", "traffic light": "红绿灯", "fire hydrant": "消防栓", "stop sign": "停车标志", "parking meter": "停车计时器",
      "bench": "长椅", "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊", "cow": "牛", "elephant": "大象", "bear": "熊",
      "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "行李箱",
      "frisbee": "飞盘", "skis": "滑雪板", "snowboard": "单板", "sports ball": "球", "kite": "风筝", "baseball bat": "球棒",
      "baseball glove": "手套", "skateboard": "滑板", "surfboard": "冲浪板", "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "酒杯",
      "cup": "杯子", "fork": "叉子", "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
      "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈", "cake": "蛋糕",
      "chair": "椅子", "couch": "沙发", "potted plant": "盆栽", "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
      "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
      "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱", "book": "书", "clock": "时钟", "vase": "花瓶",
      "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机", "toothbrush": "牙刷"}
proc = AutoProcessor.from_pretrained(MD)
proc.image_processor.size = {"longest_edge": 448 * 448, "shortest_edge": 64 * 64}
tok = proc.tokenizer
model = Qwen3VLForConditionalGeneration.from_pretrained(MD, dtype=torch.bfloat16, device_map="cuda").eval()
lm = model.model.language_model
if a.no_deepstack:
    lm._deepstack_process = lambda h, m, v: h
img_tok = model.config.image_token_id
E = lm.embed_tokens.weight  # tied with lm_head
En = torch.nn.functional.normalize(E.float(), dim=-1).to(torch.bfloat16)  # 0.6 GB instead of 1.2
L = len(lm.layers)

names, _, _ = S.coco_categories()


def target_ids(cat: str) -> list[int]:
    words = set(re.findall(r"[a-z]+", cat)) | set(SYN.get(cat, []))
    ids = set()
    for w in words:
        for v in (" " + w, w, " " + w.capitalize(), " " + w + "s", w.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    if not ids:  # e.g. " zebra" is several tokens: use the first piece, which is what would be emitted first
        ids = {tok.encode(" " + w, add_special_tokens=False)[0] for w in words}
    return sorted(ids)


TID = [target_ids(n) for n in names]
def zh_ids(n):  # first piece of the Chinese name, only if it is the whole name or >= 2 characters (else too generic)
    t = tok.encode(ZH[n], add_special_tokens=False)[0]
    piece = tok.decode([t])
    return [t] if piece == ZH[n] or len(piece) >= 2 else []


ZID = [zh_ids(n) for n in names]

# capture the residual stream entering every layer and entering the final norm
cap = {}
for i, layer in enumerate(lm.layers):
    layer.register_forward_pre_hook(lambda m, args, i=i: cap.__setitem__(i, args[0].detach()))
lm.norm.register_forward_pre_hook(lambda m, args: cap.__setitem__(L, args[0].detach()))

paths = [Path(x) for x in Path(a.image_list).read_text().split()][: a.max_images]
rng = np.random.default_rng(0)
K = L + 1
hit_true = np.zeros(K); hit_shuf = np.zeros(K); n_obj = 0
hit_zh = np.zeros(K); hit_zh_shuf = np.zeros(K); hit_any = np.zeros(K); cjk_top1 = np.zeros(K)
n_zh = 0
rank_true = [[] for _ in range(K)]
wordlike_img = np.zeros(K); wordlike_txt = np.zeros(K); norm_img = np.zeros(K); norm_txt = np.zeros(K)
n_img_tok = n_txt_tok = 0
top1_bg_content = np.zeros(K)
maps, tops, t0 = {}, {}, time.time()

for k, path in enumerate(paths):
    try:
        img = Image.open(path).convert("RGB")
    except OSError as e:  # a few COCO downloads are truncated
        print("skip", path.name, e, flush=True)
        continue
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    _, h, w = inp["image_grid_thw"][0].tolist()
    gh, gw = h // 2, w // 2
    meta = dict(stem=path.stem, gh=gh, gw=gw, W=img.width, H=img.height)
    lab = S.token_labels([meta])[0]
    cap.clear()
    with torch.no_grad():
        model(**inp)
    st = dict(cap)  # snapshot: our own lm.norm() calls below re-trigger the norm hook
    ids = inp["input_ids"][0]
    im_pos = (ids == img_tok).nonzero().squeeze(1)
    tx_pos = (ids != img_tok).nonzero().squeeze(1)
    assert len(im_pos) == gh * gw, (len(im_pos), gh, gw)
    obj = np.flatnonzero(lab >= 0)
    shuf = [rng.choice([c for c in range(80) if c != lab[o] and TID[c]]) for o in obj]
    shuf_zh = [rng.choice([c for c in range(80) if c != lab[o] and ZID[c]]) for o in obj]
    is_test = path.stem in TESTS
    if is_test:
        tc = names.index(TESTS[path.stem])
        maps[path.stem] = dict(prob=np.zeros((K, gh * gw), np.float32), lab=lab, gh=gh, gw=gw, cat=TESTS[path.stem])
        tops[path.stem] = {}
    with torch.no_grad():
        for l in range(K):
            hs = st[l][0]
            hi, ht = hs[im_pos].float(), hs[tx_pos].float()
            norm_img[l] += hi.norm(dim=-1).sum().item(); norm_txt[l] += ht.norm(dim=-1).sum().item()
            wordlike_img[l] += (torch.nn.functional.normalize(hi, dim=-1).to(torch.bfloat16) @ En.T).float().max(-1).values.sum().item()
            wordlike_txt[l] += (torch.nn.functional.normalize(ht, dim=-1).to(torch.bfloat16) @ En.T).float().max(-1).values.sum().item()
            logits = model.lm_head(lm.norm(hs[im_pos])).float()  # [N, V]
            top10 = logits.topk(10, dim=-1).indices.cpu().numpy()
            for j, o in enumerate(obj):
                hit_true[l] += bool(set(top10[o]) & set(TID[lab[o]]))
                hit_shuf[l] += bool(set(top10[o]) & set(TID[shuf[j]]))
                ten = set(top10[o])
                if ZID[lab[o]]:
                    zs = shuf_zh[j]
                    hit_zh[l] += bool(ten & set(ZID[lab[o]]))
                    hit_zh_shuf[l] += bool(ten & set(ZID[zs]))
                    n_zh += l == 0
                hit_any[l] += bool(ten & (set(ZID[lab[o]]) | set(TID[lab[o]])))
                tl = logits[o, TID[lab[o]]].max()
                rank_true[l].append(int((logits[o] > tl).sum().item()))
            cjk_top1[l] += sum(any("\u4e00" <= ch <= "\u9fff" for ch in tok.decode([t])) for t in top10[:, 0])
            if is_test:
                pr = logits.softmax(-1)[:, TID[tc]].sum(-1)
                maps[path.stem]["prob"][l] = pr.cpu().numpy()
                cnt = np.bincount(top10[:, 0], minlength=1)
                tops[path.stem][l] = [(tok.decode([t]), int(c)) for t, c in
                                      sorted(((t, c) for t, c in enumerate(cnt) if c), key=lambda x: -x[1])[:8]]
    n_obj += len(obj); n_img_tok += len(im_pos); n_txt_tok += len(tx_pos)
    if is_test and k < 6:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        tops[path.stem]["generated"] = tok.decode(out[0, ids.shape[0]:], skip_special_tokens=True)
    if (k + 1) % 25 == 0:
        el = time.time() - t0
        print(f"{k + 1}/{len(paths)}  {el / (k + 1):.2f}s/img  hit@10 by layer: "
              f"{np.round(hit_true / max(n_obj, 1), 3)[::4].tolist()}", flush=True)

res = dict(
    tag=a.tag, deepstack=not a.no_deepstack, n_images=len(paths), n_obj_tokens=n_obj, layers=K,
    hit10=(hit_true / n_obj).tolist(), hit10_shuffled=(hit_shuf / n_obj).tolist(),
    median_rank=[float(np.median(r)) for r in rank_true],
    top100_frac=[float(np.mean(np.array(r) < 100)) for r in rank_true],
    wordlike_img=(wordlike_img / n_img_tok).tolist(), wordlike_txt=(wordlike_txt / n_txt_tok).tolist(),
    norm_img=(norm_img / n_img_tok).tolist(), norm_txt=(norm_txt / n_txt_tok).tolist(),
    tops=tops,
    hit10_zh=(hit_zh / n_zh).tolist(), hit10_zh_shuffled=(hit_zh_shuf / n_zh).tolist(), n_zh_tokens=n_zh,
    zh_categories=[n for n in names if ZID[names.index(n)]],
    hit10_any=(hit_any / n_obj).tolist(), cjk_top1_frac=(cjk_top1 / n_img_tok).tolist(),
)
json.dump(res, open(S.AN / f"results/lens_{a.tag}.json", "w"))
np.savez(S.AN / f"lens_maps_{a.tag}.npz", **{f"{s}_{k}": v for s, m in maps.items() for k, v in m.items() if k != "cat"})
print("hit@10", np.round(res["hit10"], 3).tolist())
print("shuffled", np.round(res["hit10_shuffled"], 3).tolist())
print("median rank", res["median_rank"])
print("hit@10 zh", np.round(res["hit10_zh"], 3).tolist())
print("zh shuffled", np.round(res["hit10_zh_shuffled"], 3).tolist())
print("hit@10 en|zh", np.round(res["hit10_any"], 3).tolist())
print("CJK top-1 share", np.round(res["cjk_top1_frac"], 3).tolist())
print("wordlike img/txt", np.round(res["wordlike_img"], 3).tolist(), np.round(res["wordlike_txt"], 3).tolist())
for s, t in tops.items():
    print(s, t.get("generated", ""))
    for l in (0, 4, 8, 12, 16, 20, 24, 28):
        print(f"   L{l:2d}", t[l][:6])
