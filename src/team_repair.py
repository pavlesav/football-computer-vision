"""
Kit-hue audit of track-level team labels over a persisted game state.

Purpose: track-level team labels are ~90-95% right, so the per-team kit-colour
consensus is reliable — a track whose torso colour *consistently* disagrees
with its assigned team's consensus is a classifier error; a track whose own
samples *disagree with each other* is a BoT-SORT ID swap or occlusion case.

Three findings from building this against real data (July 2026):

* **Per-sample voting, never mean-histogram voting.** A track occluded by an
  opponent in some samples has a contaminated mean; voting per sample separates
  "consistently wrong label" (auto-fixable) from "mixed evidence" (human
  review). Verified on sut-mla #2316 (GK, consistent → flip), #86 (ID swap,
  mixed → suspect), dec-mla #2784 (occlusion, mixed → no flip).
* **Gate on kit separability.** When the two kits are not hue-separable
  (bud-sut: white vs white-blue stripes, Bhattacharyya separation 0.067 vs
  sut-mla's 0.51) the margins are pure noise — the audit refuses to propose
  anything rather than flipping randomly. Do not normalise margins by the
  separation instead; that inflates noise exactly when kits are similar.
* **Auto-applying flips changed zero events** on the validated 4-min windows
  (flipped tracks never carried the ball there) — this is artifact hygiene and
  correction-UI triage, not an event-accuracy lever. Run it before long
  batches; feed ``suspect_mixed`` rows to the human correction workflow.

Usage::

    python -m src.team_repair --match sut-mla            # report only
    python -m src.team_repair --match sut-mla --apply    # rewrite players.parquet
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .game_state import GameState, game_state_dir, _to_parquet

H_BINS, S_BINS = 18, 4
MIN_SEPARATION = 0.2      # below this the kits are not hue-separable — abstain
FLIP_MARGIN = 0.15        # per-sample |d0-d1| needed to count as decisive
FLIP_FRAC = 0.8           # fraction of samples that must prefer the other team


def torso_hist(img: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Grass-masked HS histogram of the torso region of a player bbox."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 16:
        return None
    crop = img[max(int(y1 + 0.15 * h), 0):int(y1 + 0.50 * h),
               max(int(x1 + 0.25 * w), 0):int(x1 + 0.75 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ok = ~((Hc >= 30) & (Hc <= 90) & (Sc >= 60)) & (Vc > 30)
    if ok.sum() < 40:
        return None
    hist = cv2.calcHist([hsv], [0, 1], ok.astype(np.uint8),
                        [H_BINS, S_BINS], [0, 180, 0, 256]).flatten()
    s = hist.sum()
    return hist / s if s > 0 else None


def _bh(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two normalised histograms (0 = same)."""
    return float(1.0 - np.sum(np.sqrt(a * b)))


def audit_teams(gs: GameState, video_path=None, samples_per_track: int = 5,
                min_track_frames: int = 10) -> tuple[pd.DataFrame, float]:
    """Audit every track's team label against the kit-colour consensus.

    Returns ``(table, separation)``. The table has one row per audited track:
    ``track_id, label, frac_other, med_margin, frames, status`` with status in
    {ok, flip, suspect_mixed}. Empty table when kits are not hue-separable.
    """
    video_path = video_path or Config.MATCH_VIDEOS[gs.slug]
    pl = gs.players
    lengths = pl.groupby("track_id").size()
    plx = pl[pl.track_id.isin(lengths[lengths >= min_track_frames].index)]

    picks = []
    for tid, grp in plx.groupby("track_id"):
        grp = grp.sort_values("frame")
        idx = np.linspace(0, len(grp) - 1,
                          min(samples_per_track, len(grp))).astype(int)
        picks.append(grp.iloc[idx])
    picks = pd.concat(picks).sort_values("frame")

    cap = cv2.VideoCapture(str(video_path))
    samples: dict[int, list] = {}
    cur, img = -1, None
    for r in picks.itertuples(index=False):
        f = int(r.frame)
        if f != cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            cur = f
        h = torso_hist(img, (r.x1, r.y1, r.x2, r.y2))
        if h is not None:
            samples.setdefault(int(r.track_id), []).append(h)
    cap.release()

    labels = pl.groupby("track_id")["team_id"].first()
    ref = {}
    for team in (0, 1):
        pairs = [(np.mean(hh, 0), lengths[t]) for t, hh in samples.items()
                 if labels.get(t) == team]
        if not pairs:
            return pd.DataFrame(), 0.0
        M = np.stack([p[0] for p in pairs])
        w = np.array([p[1] for p in pairs], float)
        ref[team] = (M * w[:, None]).sum(0) / w.sum()

    sep = _bh(ref[0], ref[1])
    if sep < MIN_SEPARATION:
        return pd.DataFrame(), sep

    rows = []
    for tid, hh in samples.items():
        lab = labels.get(tid)
        if lab not in (0, 1) or len(hh) < 3:
            continue
        votes = np.array([0 if _bh(h, ref[0]) < _bh(h, ref[1]) else 1
                          for h in hh])
        margins = np.array([abs(_bh(h, ref[0]) - _bh(h, ref[1])) for h in hh])
        other = 1 - lab
        frac_other = float((votes == other).mean())
        status = "ok"
        if frac_other >= FLIP_FRAC and np.median(margins) >= FLIP_MARGIN:
            status = "flip"
        elif ((votes == other) & (margins >= FLIP_MARGIN)).sum() >= 1 \
                and frac_other >= 0.1:
            status = "suspect_mixed"
        rows.append({"track_id": tid, "label": int(lab),
                     "frac_other": round(frac_other, 2),
                     "med_margin": round(float(np.median(margins)), 3),
                     "frames": int(lengths[tid]), "status": status})
    table = pd.DataFrame(rows).sort_values(
        ["status", "frames"], ascending=[True, False])
    return table, sep


def apply_flips(gs: GameState, table: pd.DataFrame) -> int:
    """Flip 'flip'-status track labels in players.parquet. Returns rows changed."""
    fl = table[table.status == "flip"]
    if fl.empty:
        return 0
    flip_map = dict(zip(fl.track_id, 1 - fl.label))
    players = gs.players.copy()
    m = players.track_id.isin(flip_map)
    players.loc[m, "team_id"] = players.loc[m, "track_id"].map(flip_map)
    _to_parquet(players, game_state_dir(gs.slug) / "players.parquet")
    return int(m.sum())


def main():
    ap = argparse.ArgumentParser(description="Kit-hue audit of team labels")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite players.parquet with 'flip' corrections")
    args = ap.parse_args()

    gs = GameState.load(args.match)
    table, sep = audit_teams(gs)
    print(f"kit separation: {sep:.3f}")
    if table.empty:
        print("kits not hue-separable on this match - no audit possible")
        return
    fl = table[table.status == "flip"]
    su = table[table.status == "suspect_mixed"]
    print(f"tracks audited: {len(table)} | flips: {len(fl)} "
          f"({int(fl.frames.sum())} player-frames) | ID-swap/occlusion "
          f"suspects: {len(su)} ({int(su.frames.sum())} player-frames)")
    show = table[table.status != "ok"]
    if len(show):
        print(show.head(25).to_string(index=False))
    if args.apply:
        n = apply_flips(gs, table)
        print(f"applied: {n} player-rows relabelled")


if __name__ == "__main__":
    main()
