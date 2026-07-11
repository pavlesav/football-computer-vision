"""
VLM jersey-number reader (Qwen2-VL-2B) — a stronger per-crop reader than easyocr
for back-of-shirt numbers.

Why: easyocr's text detector misses most legible-but-small shirt numbers (its
recall, not the vote thresholds, is what caps numbered-identity coverage). A
head-to-head on the sut-mla p2 ground-truth tracks read **6/8** with Qwen2-VL vs
**3/8** with easyocr — recovering numbers easyocr scored zero on. The wrong reads
a VLM does produce are filtered by the SAME majority-vote + structural gates
(corroboration / conflict veto / single-support), which are left completely
unchanged — only the per-crop reader swaps.

This writes the identical ``jersey_numbers.json`` schema as :mod:`src.jersey_ocr`
(``events`` auto-loads it, ``jersey_ocr --revote`` / ``--eval`` operate on it),
so it is a drop-in alternative reader. It reuses ``jersey_ocr``'s crop plan,
aggregation, and gates verbatim.

Design note: because the Phase-3 ReID cross-cut link spreads ONE confident number
to all of a player's fragments, we don't need a read on every crop — a leaner
crop budget aimed at the tallest boxes is enough, and each confident number then
propagates. Biggest marginal value on matches with NO human identity file (the
only automatic number source there).

Run::

    python -m src.jersey_vlm --match sut-mla --half 2
    python -m src.jersey_vlm --match sut-mla --half 2 --max_crops 8000
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .game_state import GameState, available_periods
from . import jersey_ocr as J

VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VLM_MIN_BOX_H = 90            # same legibility floor as easyocr path
VLM_MAX_CROPS = 22000        # full-half plans measure 18-20k; VLM reads are
                              # 17 ms/crop (RTX 5070, batch 12) so video decode
                              # dominates the run — a lean budget saves nothing
VLM_BATCH = 12
VLM_UPSCALE = 3.0
PROMPT = ("What number is printed on the back of this soccer player's shirt? "
          "Answer with ONLY the number (1-99), or 'none' if you cannot read one.")

_MODEL = None
_PROC = None


def _model():
    global _MODEL, _PROC
    if _MODEL is None:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        _MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
        _MODEL.eval()
        _PROC = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        _PROC.tokenizer.padding_side = "left"   # correct batched generation
    return _MODEL, _PROC


def _parse(text: str) -> Optional[int]:
    m = re.search(r"\d+", text)
    if not m or len(m.group()) > 2:   # "100" must not parse as 10
        return None
    n = int(m.group())
    return n if 1 <= n <= 99 else None


def read_numbers_batch(crops: list) -> list:
    """Batched VLM read: list of BGR crops -> list of Optional[int]."""
    import torch
    from PIL import Image
    model, proc = _model()
    pil = []
    for c in crops:
        up = cv2.resize(c, None, fx=VLM_UPSCALE, fy=VLM_UPSCALE,
                        interpolation=cv2.INTER_CUBIC)
        pil.append(Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    one = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[one] * len(pil), images=pil, return_tensors="pt",
                  padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trim = gen[:, inputs.input_ids.shape[1]:]
    outs = proc.batch_decode(trim, skip_special_tokens=True)
    return [_parse(o) for o in outs]


def extract(slug: str, period: int, max_crops: int = VLM_MAX_CROPS,
            force: bool = False, min_box_h: int = VLM_MIN_BOX_H) -> dict:
    """VLM analogue of :func:`jersey_ocr.extract`: same plan + aggregation +
    gates + output schema, batched Qwen2-VL reader. Writes jersey_numbers.json."""
    gs = GameState.load(slug, period=period)
    out_path = J.jersey_path(slug, period)
    if out_path.exists() and not force:
        print(f"[{slug} p{period}] jersey_numbers.json exists — skipping "
              f"(--force to redo)")
        return json.loads(out_path.read_text(encoding="utf-8"))

    jobs, track_meta = J.build_plan(gs, max_crops, min_box_h=min_box_h)
    print(f"[{slug} p{period}] VLM reading {len(jobs)} crops across "
          f"{len(set(j['track_id'] for j in jobs))} tracks")
    if not jobs:
        payload = {"meta_numbers": {}, "track_reads": {},
                   "params": {**J._params(max_crops), "reader": "qwen2-vl"}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        return payload

    by_frame: dict = defaultdict(list)
    for j in jobs:
        by_frame[j["frame"]].append(j)
    frames = sorted(by_frame)

    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[slug]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    cur = frames[0]
    track_reads: dict = defaultdict(list)
    buf_crops, buf_meta = [], []          # (crop) and (tid, frame)
    done, total, t0 = 0, len(jobs), time.time()

    def flush():
        nonlocal done
        if not buf_crops:
            return
        nums = read_numbers_batch(buf_crops)
        for (tid, f), n in zip(buf_meta, nums):
            if n is not None:
                track_reads[tid].append({"number": int(n), "conf": 1.0,
                                         "frame": int(f)})
        done += len(buf_crops)
        buf_crops.clear(); buf_meta.clear()
        if done % (VLM_BATCH * 20) < VLM_BATCH:
            el = time.time() - t0
            print(f"  {done}/{total} crops ({el:.0f}s, ~{el/max(done,1)*(total-done):.0f}s left)")

    for f in frames:
        while cur < f:
            cap.grab(); cur += 1
        ok, img = cap.read(); cur += 1
        if not ok:
            print(f"  frame {f}: read failed, stopping early")
            break
        for j in by_frame[f]:
            crop = J.torso_crop(img, j["x1"], j["y1"], j["x2"], j["y2"])
            if crop.size == 0:
                done += 1
                continue
            buf_crops.append(crop); buf_meta.append((j["track_id"], f))
            if len(buf_crops) >= VLM_BATCH:
                flush()
    flush()
    cap.release()

    n_reads = sum(len(v) for v in track_reads.values())
    print(f"[{slug} p{period}] {n_reads} accepted reads across "
          f"{len(track_reads)} tracks ({time.time()-t0:.0f}s)")

    track_reads_json = {
        str(tid): {"meta_id": int(track_meta.get(tid, tid)),
                   "reads": reads}
        for tid, reads in track_reads.items()}
    meta_numbers = J.aggregate(dict(track_reads), gs)
    payload = {
        "meta_numbers": {str(k): v for k, v in meta_numbers.items()},
        "track_reads": track_reads_json,
        "params": {**J._params(max_crops), "reader": "qwen2-vl",
                   "min_box_h": min_box_h},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[{slug} p{period}] {len(meta_numbers)} confident meta numbers "
          f"-> {out_path}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, choices=[1, 2])
    ap.add_argument("--max_crops", type=int, default=VLM_MAX_CROPS)
    ap.add_argument("--min_box_h", type=int, default=VLM_MIN_BOX_H,
                    help="crop-eligibility height floor (px); easyocr needed "
                         "90, the VLM may read smaller")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    periods = [args.half] if args.half else available_periods(args.match)
    for p in periods:
        extract(args.match, p, max_crops=args.max_crops, force=args.force,
                min_box_h=args.min_box_h)


if __name__ == "__main__":
    main()
