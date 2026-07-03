"""
Jersey-number OCR: read shirt numbers from stored player bboxes, vote them up
to meta-track level, for a **period-independent player identity**.

Why this exists: BoT-SORT track ids restart every half, and the persisted
team-classifier ResNet18 embeddings were measured to carry no within-team
identity signal (see CLAUDE.md) — so cross-half outfield ReID cannot be built
on appearance similarity. Shirt numbers are printed identity: same team + same
number in p1 and p2 = same player, regardless of how tracking fragmented them.

Numbers are on the BACK of shirts, so most crops (anyone facing the camera)
read nothing — that's expected. Reads are collected per track, voted up to
the meta-track level (:mod:`src.identity`), and only kept when the vote is
both large enough (``votes``) and consistent enough (``agreement``) — a wrong
confident number is worse than an abstention, so gates are conservative by
design and are never loosened to raise coverage.

Usage::

    python -m src.jersey_ocr --match sut-mla --half 2
    python -m src.jersey_ocr --match sut-mla --half 2 --eval
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import identity as identity_mod
from .config import Config
from .game_state import GameState, available_periods, game_state_dir

# ── Tunables ──────────────────────────────────────────────────────────────
MIN_TRACK_ROWS = 25           # ignore flicker tracks entirely
MIN_BOX_H = 90                 # px; below this digits aren't legible
MAX_FRAMES_PER_TRACK = 12
FRAME_DEDUP_GAP = 25           # frames; picks closer than this to an already
                                # -picked frame are skipped (spread in time)
UPSCALE = 4.0
OCR_CONF_MIN = 0.5
VOTE_MIN = 4                   # >=3 let a single lucky/wrong track (e.g. one
                                # merged into the wrong meta by consolidation)
                                # ship a confident number off 3 misattributed
                                # reads alone (measured: bok-jed track 3, 3/3
                                # "26" reads, inherited by track 315's meta)
AGREEMENT_MIN = 0.6
# A full half has ~4300-5600 tracks and ~1500-1600 metas, but only ~20-21k
# crops total across EVERY eligible track's up-to-12 frames (measured on
# sut-mla p1/p2) — meta-first ordering means a small budget starves nearly
# everything but the top handful of "mega-metas" (a heavily-fragmented
# prominent player can alone own 100+ tracks). 3000 measured 0/8 ground-truth
# tracks even attempted (all ranked below the metas the budget reached).
DEFAULT_MAX_CROPS = 22000

# Torso crop geometry (fraction of bbox h/w) — numbers sit on the upper back;
# verified visually on sut-mla p2 ground-truth tracks (contact sheet QC).
CROP_Y0, CROP_Y1 = 0.12, 0.55
CROP_X_MARGIN = 0.15

_READER = None


def _reader():
    """easyocr on CPU, single shared instance (a few thousand small crops
    doesn't need the GPU, which perception runs claim)."""
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


# ── Crop + OCR ────────────────────────────────────────────────────────────

def torso_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = y2 - y1, x2 - x1
    cy0 = max(int(y1 + CROP_Y0 * h), 0)
    cy1 = int(y1 + CROP_Y1 * h)
    cx0 = max(int(x1 + CROP_X_MARGIN * w), 0)
    cx1 = int(x2 - CROP_X_MARGIN * w)
    return img[cy0:cy1, cx0:cx1]


def read_number(crop: np.ndarray) -> Optional[tuple]:
    """Best (number, confidence) in ``crop``, or None. Accepts only 1-2 digit
    text parsing to 1..99 at easyocr confidence >= OCR_CONF_MIN — "2" and
    "28" are different numbers, never merged as substrings."""
    if crop.size == 0:
        return None
    up = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE,
                    interpolation=cv2.INTER_CUBIC)
    res = _reader().readtext(up, allowlist="0123456789")
    best = None
    for _bbox, text, conf in res:
        t = text.strip()
        if not t.isdigit() or not (1 <= len(t) <= 2):
            continue
        n = int(t)
        if not (1 <= n <= 99) or conf < OCR_CONF_MIN:
            continue
        if best is None or conf > best[1]:
            best = (n, float(conf))
    return best


# ── Crop planning ─────────────────────────────────────────────────────────

def _pick_frames(sub: pd.DataFrame) -> list:
    """Up to MAX_FRAMES_PER_TRACK rows: tallest boxes (h >= MIN_BOX_H) first,
    skipping any candidate within FRAME_DEDUP_GAP frames of an already-picked
    one so the sample spreads across the track's lifetime."""
    tall = sub[sub.h >= MIN_BOX_H].sort_values("h", ascending=False)
    picks, picked_frames = [], []
    for r in tall.itertuples(index=False):
        f = int(r.frame)
        if any(abs(f - pf) < FRAME_DEDUP_GAP for pf in picked_frames):
            continue
        picks.append(r)
        picked_frames.append(f)
        if len(picks) >= MAX_FRAMES_PER_TRACK:
            break
    return picks


def build_plan(gs: GameState, max_crops: int) -> tuple:
    """Crop jobs ordered by descending meta total-frames (spend the crop
    budget on the most prominent players first), truncated to ``max_crops``.
    Returns ``(jobs, track_meta)`` — ``jobs``: list of dicts with
    track_id/meta_id/frame/x1..y2; ``track_meta``: {track_id: meta_id} used
    for this run (tracks absent from :func:`identity.meta_map` — e.g.
    close-up-only tracks with no pitch-valid rows — fall back to being their
    own singleton meta, same convention :mod:`src.events` already uses)."""
    row_counts = gs.players.groupby("track_id").size()
    eligible = set(int(t) for t in row_counts[row_counts >= MIN_TRACK_ROWS].index)
    if not eligible:
        return [], {}

    meta_of = identity_mod.meta_map(gs)
    track_meta = {tid: int(meta_of.get(tid, tid)) for tid in eligible}

    meta_rows: dict = defaultdict(int)
    tracks_by_meta: dict = defaultdict(list)
    for tid in eligible:
        mid = track_meta[tid]
        meta_rows[mid] += int(row_counts[tid])
        tracks_by_meta[mid].append(tid)
    meta_order = sorted(meta_rows, key=lambda m: -meta_rows[m])

    pl = gs.players[gs.players.track_id.isin(eligible)].copy()
    pl["h"] = pl.y2 - pl.y1
    by_track = {tid: g for tid, g in pl.groupby("track_id")}

    jobs = []
    for mid in meta_order:
        for tid in sorted(tracks_by_meta[mid], key=lambda t: -row_counts[t]):
            for r in _pick_frames(by_track[tid]):
                jobs.append({"track_id": int(tid), "meta_id": int(mid),
                            "frame": int(r.frame), "x1": int(r.x1),
                            "y1": int(r.y1), "x2": int(r.x2), "y2": int(r.y2)})
                if len(jobs) >= max_crops:
                    return jobs, track_meta
    return jobs, track_meta


# ── Voting / aggregation ──────────────────────────────────────────────────

def _vote(track_reads: dict, track_meta: dict) -> tuple:
    """Collapse per-track reads to per-meta plurality votes. Returns
    ``(meta_numbers, track_reads_json)``."""
    meta_nums: dict = defaultdict(list)
    track_reads_json = {}
    for tid, reads in track_reads.items():
        mid = track_meta.get(tid, tid)
        track_reads_json[str(tid)] = {
            "meta_id": int(mid),
            "reads": [{"number": r["number"], "conf": round(r["conf"], 3),
                      "frame": r["frame"]} for r in reads],
        }
        meta_nums[mid].extend(r["number"] for r in reads)

    meta_numbers = {}
    for mid, nums in meta_nums.items():
        if not nums:
            continue
        counts = Counter(nums)
        number, votes = counts.most_common(1)[0]
        total = len(nums)
        agreement = votes / total
        if votes >= VOTE_MIN and agreement >= AGREEMENT_MIN:
            meta_numbers[mid] = {"number": int(number), "votes": int(votes),
                                 "reads": int(total),
                                 "agreement": round(agreement, 3)}
    return meta_numbers, track_reads_json


def resolve_uniqueness(meta_numbers: dict, meta_team: dict) -> dict:
    """Within (team, number), keep only the highest-vote meta; demote the
    rest. Metas whose team is unknown (absent from consolidation — no
    pitch-valid rows) are left as-is, since they can't be safely grouped."""
    groups: dict = defaultdict(list)
    for mid, rec in meta_numbers.items():
        team = meta_team.get(mid)
        if team is None:
            continue
        groups[(team, rec["number"])].append(mid)

    out = dict(meta_numbers)
    for (team, num), mids in groups.items():
        if len(mids) <= 1:
            continue
        ranked = sorted(mids, key=lambda m: -out[m]["votes"])
        keep = ranked[0]
        for m in ranked[1:]:
            print(f"  uniqueness collision: team {team} #{num} - keeping "
                  f"meta {keep} ({out[keep]['votes']} votes), demoting meta "
                  f"{m} ({out[m]['votes']} votes)")
            del out[m]
    return out


def id_swap_suspects(track_reads: dict, top_n: int = 10) -> list:
    """Tracks whose reads split between >=2 distinct numbers, most-split
    first — surfaced as a diagnostic only; nothing is auto-split."""
    suspects = []
    for tid, reads in track_reads.items():
        nums = [r["number"] for r in reads]
        if len(set(nums)) < 2:
            continue
        counts = Counter(nums)
        total = sum(counts.values())
        _top, top_votes = counts.most_common(1)[0]
        minority_frac = (total - top_votes) / total
        suspects.append((tid, dict(counts), minority_frac))
    suspects.sort(key=lambda s: -s[2])
    return suspects[:top_n]


# ── Extraction ────────────────────────────────────────────────────────────

def jersey_path(slug: str, period: int) -> Path:
    return game_state_dir(slug, period) / "jersey_numbers.json"


def load_jersey_numbers(slug: str, period: int) -> Optional[dict]:
    """{meta_id: {"number", "votes", "reads", "agreement"}} for confident
    metas, or None if extraction hasn't been run for this (slug, period)."""
    p = jersey_path(slug, period)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {int(k): v for k, v in d.get("meta_numbers", {}).items()}


def extract(slug: str, period: int, max_crops: int = DEFAULT_MAX_CROPS,
           force: bool = False) -> dict:
    gs = GameState.load(slug, period=period)
    out_path = jersey_path(slug, period)
    if out_path.exists() and not force:
        print(f"[{slug} p{period}] jersey_numbers.json already exists - "
              f"skipping (--force to redo)")
        return json.loads(out_path.read_text())

    jobs, track_meta = build_plan(gs, max_crops)
    n_tracks = len(set(j["track_id"] for j in jobs))
    n_metas = len(set(j["meta_id"] for j in jobs))
    print(f"[{slug} p{period}] {len(jobs)} crops planned across {n_tracks} "
          f"tracks, {n_metas} metas (budget {max_crops})")
    if not jobs:
        payload = {"meta_numbers": {}, "track_reads": {},
                  "params": _params(max_crops)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        return payload

    by_frame: dict = defaultdict(list)
    for j in jobs:
        by_frame[j["frame"]].append(j)
    frames = sorted(by_frame)

    video = str(Config.MATCH_VIDEOS[slug])
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    cur = frames[0]

    track_reads: dict = defaultdict(list)
    total = len(jobs)
    done = 0
    t0 = time.time()
    for f in frames:
        while cur < f:
            cap.grab()
            cur += 1
        ok, img = cap.read()
        cur += 1
        if not ok:
            print(f"  frame {f}: video read failed, stopping early "
                  f"({done}/{total} crops processed)")
            break
        for j in by_frame[f]:
            crop = torso_crop(img, j["x1"], j["y1"], j["x2"], j["y2"])
            res = read_number(crop)
            if res is not None:
                track_reads[j["track_id"]].append(
                    {"number": res[0], "conf": res[1], "frame": f})
            done += 1
        if done % 200 < len(by_frame[f]):
            elapsed = time.time() - t0
            rate = elapsed / max(done, 1)
            eta = rate * (total - done)
            print(f"  {done}/{total} crops ({elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining)")
    cap.release()

    n_reads = sum(len(v) for v in track_reads.values())
    print(f"[{slug} p{period}] {n_reads} accepted reads across "
          f"{len(track_reads)} tracks with >=1 read "
          f"({time.time()-t0:.0f}s total)")

    meta_numbers, track_reads_json = _vote(track_reads, track_meta)
    meta_team = identity_mod.meta_teams(gs)
    n_before = len(meta_numbers)
    meta_numbers = resolve_uniqueness(meta_numbers, meta_team)
    if len(meta_numbers) != n_before:
        print(f"[{slug} p{period}] uniqueness resolution: {n_before} -> "
              f"{len(meta_numbers)} confident metas")

    suspects = id_swap_suspects(track_reads)
    if suspects:
        print(f"\n[{slug} p{period}] top ID-swap suspects (split reads, "
              f"NOT auto-split):")
        for tid, counts, frac in suspects:
            print(f"  track {tid}: {counts} (minority fraction {frac:.2f})")

    payload = {
        "meta_numbers": {str(k): v for k, v in meta_numbers.items()},
        "track_reads": track_reads_json,
        "params": _params(max_crops),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[{slug} p{period}] {len(meta_numbers)} confident meta numbers "
          f"-> {out_path}")
    return payload


def _params(max_crops: int) -> dict:
    return {"min_track_rows": MIN_TRACK_ROWS, "min_box_h": MIN_BOX_H,
           "max_frames_per_track": MAX_FRAMES_PER_TRACK,
           "frame_dedup_gap": FRAME_DEDUP_GAP, "upscale": UPSCALE,
           "ocr_conf_min": OCR_CONF_MIN, "vote_min": VOTE_MIN,
           "agreement_min": AGREEMENT_MIN, "max_crops": max_crops}


# ── Evaluation ────────────────────────────────────────────────────────────

GROUND_TRUTH = {
    ("sut-mla", 2): [
        (42894, 99, ""),
        (43265, 5, "yellow player until ~f146404, track then swaps onto a "
                   "blue player - ID swap; per-track votes may split"),
        (43355, 5, "same physical player as 43265's yellow phase"),
        (43486, 28, ""),
        (43634, 19, ""),
        (43672, 3, ""),
        (42973, 3, "likely the same player as 43672 (different track)"),
        (43203, 77, "medium confidence"),
    ],
    ("bok-jed", 1): [
        (192, 9, ""),
        (146, 8, ""),
        (129, 30, ""),
        (150, 10, ""),
        (315, 14, "medium confidence"),
    ],
}


def evaluate(slug: str, period: int) -> None:
    gt = GROUND_TRUTH.get((slug, period))
    if not gt:
        print(f"No ground truth table for {slug} p{period}")
        return
    jn = load_jersey_numbers(slug, period)
    if jn is None:
        print(f"No jersey_numbers.json for {slug} p{period} - run "
              f"extraction first (python -m src.jersey_ocr --match {slug} "
              f"--half {period})")
        return
    gs = GameState.load(slug, period=period)
    tmap = identity_mod.meta_map(gs)

    print(f"\n=== {slug} p{period}: ground-truth accuracy ===")
    n_assigned = n_correct = 0
    for tid, expected, note in gt:
        mid = int(tmap.get(tid, tid))
        rec = jn.get(mid)
        if rec is None:
            print(f"  track {tid} (meta {mid}): ABSTAIN (expected {expected}) "
                  f"{note}")
            continue
        n_assigned += 1
        got = rec["number"]
        ok = got == expected
        n_correct += int(ok)
        print(f"  track {tid} (meta {mid}): got {got}, expected {expected} "
              f"[{'OK' if ok else 'WRONG'}] votes={rec['votes']} "
              f"agreement={rec['agreement']:.2f} {note}")
    if n_assigned == 0:
        print(f"assigned 0/{len(gt)} - nothing to grade (abstaining is "
              f"acceptable; the gate is about wrong answers, not coverage)")
    else:
        acc = n_correct / n_assigned
        print(f"assigned {n_assigned}/{len(gt)}, correct {n_correct}/"
              f"{n_assigned} = {acc*100:.1f}% (gate: >=90%)")

    cons = identity_mod.consolidate_tracks(gs)
    meta_team = dict(zip(cons.meta_id.astype(int), cons.team.astype(int)))
    groups: dict = defaultdict(list)
    for mid, rec in jn.items():
        team = meta_team.get(mid)
        if team is None:
            continue
        groups[(team, rec["number"])].append(mid)
    violations = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\nuniqueness violations: {len(violations)} (gate: 0)")
    for (team, num), mids in violations.items():
        print(f"  team {team} number {num}: metas {mids}")


def cross_half_check(slug: str) -> None:
    periods = available_periods(slug)
    if len(periods) < 2:
        print(f"\ncross-half check: {slug} has only {len(periods)} "
              f"period(s) - skip")
        return
    p1, p2 = periods[0], periods[1]
    jn1, jn2 = load_jersey_numbers(slug, p1), load_jersey_numbers(slug, p2)
    if jn1 is None or jn2 is None:
        print(f"\ncross-half check: missing jersey_numbers.json for p{p1} "
              f"or p{p2} - skip")
        return
    gs1, gs2 = GameState.load(slug, period=p1), GameState.load(slug, period=p2)
    mt1 = identity_mod.meta_teams(gs1)
    mt2 = identity_mod.meta_teams(gs2)

    print(f"\n=== {slug}: cross-half consistency ===")
    for team in (0, 1):
        nums1 = {rec["number"] for mid, rec in jn1.items() if mt1.get(mid) == team}
        nums2 = {rec["number"] for mid, rec in jn2.items() if mt2.get(mid) == team}
        overlap = nums1 & nums2
        smaller = min(len(nums1), len(nums2)) or 1
        frac = len(overlap) / smaller
        print(f"  team {team}: p{p1} {sorted(nums1)}  p{p2} {sorted(nums2)}")
        print(f"    matched {sorted(overlap)} = {len(overlap)}/{smaller} "
              f"({frac*100:.0f}% of smaller set, gate: >=60%)")


def main():
    ap = argparse.ArgumentParser(
        description="Jersey-number OCR extraction + evaluation")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--half", type=int, required=True, choices=[1, 2])
    ap.add_argument("--max_crops", type=int, default=DEFAULT_MAX_CROPS)
    ap.add_argument("--force", action="store_true",
                    help="redo extraction even if jersey_numbers.json exists")
    ap.add_argument("--eval", action="store_true",
                    help="evaluate an existing jersey_numbers.json instead "
                         "of extracting")
    args = ap.parse_args()

    if args.eval:
        evaluate(args.match, args.half)
        cross_half_check(args.match)
        return

    extract(args.match, args.half, max_crops=args.max_crops, force=args.force)


if __name__ == "__main__":
    main()
