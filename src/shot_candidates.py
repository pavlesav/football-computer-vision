"""
Shot-candidate detector: surface *moments that might be shots* from the ball
track, for a human to confirm-and-tag. The product cannot resell SofaScore's
shots (and they arrive days late), so shots must come from the broadcast +
a human — but the human should only ever review a short candidate list, not
scrub 90 minutes.

Design posture: **high recall, tolerant precision.** Missing a shot is
expensive (the human never sees it); a false candidate is cheap (one click to
reject in the tagging UI). So the gate is deliberately loose — any fast ball
heading at a goal mouth from the attacking third — and we measure recall
against SofaScore's shot timestamps to prove we don't miss.

MEASURED STATUS (2026-07-07) — this is a RESEARCH STARTING POINT, not shipped.
Two signals were evaluated against SofaScore shot timestamps:

* **Ball-toward-goal** (this module's :func:`detect_period`): recall only
  13-22%. Root cause is fundamental, not tunable — shots happen exactly when
  the broadcast zooms to a close-up (every goal / big chance), where there is
  no trusted pitch-space ball position. The sut-mla goal itself produced zero
  ball candidates. Low precision too (~30 candidates for ~9 shots).
* **Camera wide→close-up transition** (:func:`camera_shot_moments`): recall
  **~83%** (sut-mla 9/9, jez-jed 13/18, mla-bud 25/30 within ±20s) — the
  director zooms in on every shot. BUT precision is ~2% (400-700 transitions
  per match; these broadcasts cut to close-up constantly). A final-third ball
  filter cut count ~2x but halved recall.

Conclusion: the camera-transition signal is the right foundation (high recall),
but needs a precision filter that does NOT depend on the ball (which isn't
tracked at the shot). The promising next step is **player-box-occupancy before
the cut** (attackers in the penalty area — players are tracked more reliably
than the ball) plus **restart type after** (a shot that misses/saves tends to
yield a goal-kick or corner). Until then, shots stay on the manual-tag path.

Run::

    python -m src.shot_candidates --match sut-mla            # ball candidates
    python -m src.shot_candidates --match sut-mla --eval     # both signals vs SofaScore
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .config import Config
from .game_state import GameState, available_periods
from .events import ball_series
from .roles import infer_attack_direction

PITCH_L, PITCH_W = 105.0, 68.0
GOAL_Y = 34.0
GOAL_HALF_W = 6.0            # goal mouth half-width + generous margin (m)
MIN_SHOT_SPEED = 11.0        # m/s toward goal to be a candidate
FINAL_THIRD_X = 70.0         # attacking third depth (mirror for left goal)
MAX_GOAL_DIST = 40.0         # only within this of the target goal
MERGE_GAP_S = 2.5            # candidate frames within this are one shot
MIN_RUN_FRAMES = 2


@dataclass
class ShotCandidate:
    period: int
    time_sec: float
    team: int
    goal_side: str            # 'left' | 'right'
    x: float
    y: float
    speed: float
    proj_goal_y: float        # where the path crosses the goal line


def _target_goal(team: int, dirs) -> str:
    """Which goal `team` attacks this period: 'right' (x=105) or 'left'."""
    return "right" if dirs.attack_ltr.get(team, True) else "left"


def detect_period(gs, period: int) -> list:
    dirs = infer_attack_direction(gs)
    # team that attacks each goal this half
    team_for = {}
    for team in (0, 1):
        team_for[_target_goal(team, dirs)] = team

    b = ball_series(gs).dropna(subset=["bx", "by", "speed"])
    cands = []
    for r in b.itertuples(index=False):
        vx, vy, sp = r.vx, r.vy, r.speed
        if not np.isfinite(sp) or sp < MIN_SHOT_SPEED:
            continue
        # test both goals; ball must head toward one from its attacking third
        for side, gx in (("right", PITCH_L), ("left", 0.0)):
            toward = (vx > 0) if side == "right" else (vx < 0)
            if not toward:
                continue
            in_third = (r.bx > FINAL_THIRD_X) if side == "right" \
                else (r.bx < PITCH_L - FINAL_THIRD_X)
            if not in_third:
                continue
            dist = abs(gx - r.bx)
            if dist > MAX_GOAL_DIST or abs(vx) < 1e-3:
                continue
            # project straight path to the goal line, check it crosses mouth
            t = (gx - r.bx) / vx
            if t < 0:
                continue
            gy = r.by + vy * t
            if abs(gy - GOAL_Y) > GOAL_HALF_W:
                continue
            cands.append((int(r.frame), float(r.time_sec), side,
                          float(r.bx), float(r.by), float(sp), float(gy),
                          team_for.get(side, -1)))
            break
    return _merge(cands, gs.fps, period)


def _merge(raw: list, fps: float, period: int) -> list:
    """Collapse consecutive candidate frames into discrete shots (keep the
    fastest frame of each run)."""
    if not raw:
        return []
    raw.sort()
    merge_gap_f = MERGE_GAP_S * fps
    runs = [[raw[0]]]
    for c in raw[1:]:
        if c[0] - runs[-1][-1][0] <= merge_gap_f:
            runs[-1].append(c)
        else:
            runs.append([c])
    out = []
    for run in runs:
        if len(run) < MIN_RUN_FRAMES:
            continue
        best = max(run, key=lambda c: c[5])   # fastest frame
        frame, tsec, side, bx, by, sp, gy, team = best
        out.append(ShotCandidate(period, round(tsec, 1), team, side,
                                 round(bx, 1), round(by, 1), round(sp, 1),
                                 round(gy, 1)))
    return out


def detect_match(slug: str) -> list:
    out = []
    for p in available_periods(slug):
        gs = GameState.load(slug, period=p)
        out.extend(detect_period(gs, p))
    return out


def camera_shot_moments(gs, period: int) -> list:
    """Wide→close-up transition times (seconds) — the broadcast director's
    zoom is an ~83%-recall shot signal (see module docstring). High recall,
    low precision; the foundation for a future box-occupancy-filtered
    detector, not a finished candidate list on its own."""
    import numpy as np
    f = gs.frames.sort_values("frame")
    wide = f.is_wide_shot.fillna(False).to_numpy()
    tsec = f.time_sec.to_numpy()
    idx = np.where((wide[:-1]) & (~wide[1:]))[0]
    return [float(tsec[i + 1]) for i in idx]


# ── Validation vs SofaScore ──────────────────────────────────────────────────

def _sofa_shots(slug: str) -> list:
    """(period, half_time_sec) for each SofaScore shot. Their timeSeconds is
    continuous match time; second-half kickoff is ~2700s nominal (± first-half
    stoppage), so p2 alignment is approximate — validated with a wide window."""
    p = Config.PROJECT_ROOT / "data" / "sofa_truth" / slug / "match_shots.csv"
    if not p.exists():
        return []
    import pandas as pd
    df = pd.read_csv(p)
    out = []
    for r in df.itertuples(index=False):
        ts = float(r.timeSeconds)
        if ts <= 2820:
            out.append((1, ts, r.shotType))
        else:
            out.append((2, ts - 2700, r.shotType))
    return out


def evaluate(slug: str, window_s: float = 25.0) -> dict:
    cands = detect_match(slug)
    shots = _sofa_shots(slug)
    by_p = {1: [], 2: []}
    for c in cands:
        by_p[c.period].append(c.time_sec)
    # camera-transition signal too
    cam = {1: [], 2: []}
    for p in available_periods(slug):
        cam[p] = camera_shot_moments(GameState.load(slug, period=p), p)
    hit = cam_hit = 0
    for per, tsec, stype in shots:
        if [t for t in by_p.get(per, []) if abs(t - tsec) <= window_s]:
            hit += 1
        if [t for t in cam.get(per, []) if abs(t - tsec) <= window_s]:
            cam_hit += 1
    n = len(shots) or 1
    n_cam = sum(len(v) for v in cam.values())
    print(f"[{slug}] BALL signal: {len(cands)} cands, recall {hit}/{len(shots)}"
          f"  |  CAMERA signal: {n_cam} transitions, recall {cam_hit}/{len(shots)}"
          f"  (±{window_s:.0f}s)")
    return {"candidates": len(cands), "shots": len(shots),
            "ball_recall": hit / n, "cam_recall": cam_hit / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.eval:
        evaluate(args.match)
        return
    for c in detect_match(args.match):
        print(f"  p{c.period} {int(c.time_sec//60):>2}:{int(c.time_sec%60):02d} "
              f"team{c.team} -> {c.goal_side} goal @ ({c.x},{c.y}) "
              f"{c.speed}m/s, crosses y={c.proj_goal_y}")


if __name__ == "__main__":
    main()
