"""
Tracklet SPLIT by jersey-read inconsistency — the missing SoccerNet-GSR step.

The dominant identity error class (confirmed independently by the human and
model-vision review sweeps: same-team number conflicts inside one group) is
an ID SWAP: BoT-SORT hands a track from one player to another mid-life, so
one track_id carries two players. Everything downstream (consolidation,
jersey voting, propagation, event attribution) then fuses the two players.

The highest-precision split signal we already store is the per-frame VLM
jersey reads (``track_reads`` in jersey_numbers.json): a track that reads
number A consistently, then number B consistently, has provably changed
player at some frame between the two runs. This module:

1. finds tracks whose reads form >= 2 temporally-ordered runs of DIFFERENT
   numbers, each run >= MIN_RUN_READS reads (the same evidence bar as the
   aggregate conflict veto);
2. splits the track in players.parquet at the midpoint between adjacent
   runs (rows after the boundary get a fresh track_id above the current
   max), with a backup written next to the parquet;
3. reassigns the jersey reads to the new track ids by frame, so a following
   ``jersey_ocr --revote`` turns the previously-vetoed conflict into TWO
   clean confident numbers.

After splitting run the standard chain: revote -> reid --force -> events.

Run::

    python -m src.track_split --match jed-ars --half 2          # dry run
    python -m src.track_split --match jed-ars --half 2 --apply
"""
from __future__ import annotations

import argparse
import json
import shutil

import pandas as pd

from .game_state import GameState, available_periods, game_state_dir
from .jersey_ocr import jersey_path

MIN_RUN_READS = 2      # a run must have this many consistent reads to count
# Fragment ids must stay above real BoT-SORT ids (~60k) and BELOW the export
# id spaces they could collide with after period-2 namespacing (+100000):
# jersey ids live at 800000-801999 and GK ids at 990000+. 900000 keeps p1
# fragments (900000+k) and p2 fragments (1000000+k after ns) unique.
SPLIT_ID_BASE = 900000


def _runs(reads: list) -> list:
    """Temporally-ordered maximal runs of identical numbers.
    Returns [(number, first_frame, last_frame, n_reads), ...]."""
    rs = sorted(reads, key=lambda r: int(r["frame"]))
    out = []
    for r in rs:
        n, f = int(r["number"]), int(r["frame"])
        if out and out[-1][0] == n:
            num, f0, _, c = out[-1]
            out[-1] = (num, f0, f, c + 1)
        else:
            out.append((n, f, f, 1))
    return out


def find_splits(slug: str, period: int) -> dict:
    """{track_id: [split_frame, ...]} for tracks whose reads prove an ID
    swap. A split lands between two adjacent runs of different numbers when
    BOTH runs carry >= MIN_RUN_READS reads (single stray reads never split)."""
    p = jersey_path(slug, period)
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    splits: dict = {}
    for tid, v in d.get("track_reads", {}).items():
        runs = [r for r in _runs(v.get("reads", [])) if r[3] >= MIN_RUN_READS]
        # merge adjacent same-number strong runs (weak runs removed between)
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                num, f0, _, c = merged[-1]
                merged[-1] = (num, f0, r[2], c + r[3])
            else:
                merged.append(list(r))
        if len(merged) < 2:
            continue
        cuts = []
        for a, b in zip(merged, merged[1:]):
            cuts.append((int(a[2]) + int(b[1])) // 2)   # midpoint frame
        splits[int(tid)] = cuts
    return splits


def apply_splits(slug: str, period: int, splits: dict) -> int:
    """Rewrite players.parquet + jersey track_reads with the split tracks.
    Returns the number of new fragments created."""
    pdir = game_state_dir(slug, period)
    ppath = pdir / "players.parquet"
    bak = pdir / "players.pre_split.parquet"
    if not bak.exists():
        shutil.copy2(ppath, bak)
    pl = pd.read_parquet(ppath)
    next_id = max(int(pl.track_id.max()) + 1, SPLIT_ID_BASE)

    jp = jersey_path(slug, period)
    jd = json.loads(jp.read_text(encoding="utf-8"))
    reads = jd.get("track_reads", {})

    n_new = 0
    id_map_log = {}
    for tid, cuts in sorted(splits.items()):
        segs = []          # (start_frame_inclusive, new_tid or tid)
        prev = None
        seg_ids = [int(tid)]
        for cut in sorted(cuts):
            seg_ids.append(next_id)
            next_id += 1
        # rows: frame < cuts[0] keep tid; cuts[i-1] <= frame < cuts[i] get
        # seg_ids[i]
        mask_track = pl.track_id == tid
        frames = pl.loc[mask_track, "frame"]
        bounds = sorted(cuts)
        seg_index = frames.apply(
            lambda f: sum(1 for c in bounds if f >= c))
        pl.loc[mask_track, "track_id"] = seg_index.map(
            lambda i: seg_ids[i]).astype(pl.track_id.dtype)
        n_new += len(bounds)
        id_map_log[tid] = seg_ids

        # move jersey reads to the segment that owns their frame
        old = reads.pop(str(tid), None)
        if old:
            per_seg: dict = {}
            for r in old.get("reads", []):
                i = sum(1 for c in bounds if int(r["frame"]) >= c)
                per_seg.setdefault(seg_ids[i], []).append(r)
            for sid, rr in per_seg.items():
                reads[str(sid)] = {"meta_id": int(sid), "reads": rr}

    pl.to_parquet(ppath)
    jd["track_reads"] = reads
    jd.setdefault("params", {})["track_split"] = {
        "min_run_reads": MIN_RUN_READS,
        "splits": {str(k): v for k, v in id_map_log.items()}}
    jp.write_text(json.dumps(jd, indent=2))
    return n_new


def main():
    ap = argparse.ArgumentParser(description="Split ID-swapped tracks at "
                                 "jersey-read conflict boundaries")
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    periods = [args.half] if args.half else available_periods(args.match)
    for p in periods:
        splits = find_splits(args.match, p)
        n_cuts = sum(len(v) for v in splits.values())
        print(f"[{args.match} p{p}] {len(splits)} tracks with proven ID "
              f"swaps ({n_cuts} split points)")
        for tid, cuts in sorted(splits.items())[:12]:
            print(f"    track {tid}: split at frames {cuts}")
        if args.apply and splits:
            n = apply_splits(args.match, p, splits)
            print(f"[{args.match} p{p}] created {n} new fragments "
                  f"(backup: players.pre_split.parquet)")
            print(f"    next: python -m src.jersey_ocr --match {args.match} "
                  f"--half {p} --revote && python -m src.reid --match "
                  f"{args.match} --half {p} --force && python -m src.events "
                  f"--match {args.match}")


if __name__ == "__main__":
    main()
