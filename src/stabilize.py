"""
Offline homography stabilization over a persisted game state.

Why: the stored per-frame P is jittery — measured on the sut-mla full half,
frame-to-frame reference-point motion has p90 ≈ 49 px of high-frequency shake,
and the breakdown shows why: fresh-PnLCalib frames jitter 3x worse (p90 90 px)
than the optical-flow frames between them (32 px). With ``--pnl_stride 3`` that
is a yank every third frame — an 8 Hz sawtooth the eye reads as "everything
wobbles". Each estimator is fine; the *sequence* was never smoothed with
hindsight.

What: a non-causal smoothing pass over the whole stored sequence (analysis
side: no GPU, no video decode, seconds to run — same philosophy as the ball
tracker). Per frame we project four fixed pitch points through the stored P,
smooth those eight coordinate trajectories in time (median filter to kill
single-frame spikes, then Savitzky-Golay to keep real pans while removing the
sawtooth), and refit an exact plane homography from the four smoothed
correspondences. Player pitch positions are then **recomputed** from their
stored image feet with the stabilized P — they carried the same jitter.

Segments are broken at camera cuts / P dropouts (no smoothing across), and the
z-column of the rebuilt P is zero: only the z=0 plane survives, which is all
events, the minimap, and the pitch-line overlay ever use.

Usage::

    python -m src.stabilize --match sut-mla            # report metrics only
    python -m src.stabilize --match sut-mla --apply    # rewrite the artifact
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

from .game_state import GameState, P_COLS, _to_parquet

# Four well-spread pitch anchors (natural metres). Exact 4-point homography
# refit is invertible for any non-degenerate quad; these sit centrally so
# their image trajectories are numerically tame on wide shots.
ANCHORS = np.array([[30.0, 14.0], [75.0, 14.0], [75.0, 54.0], [30.0, 54.0]])

CUT_PX = 250.0          # median anchor displacement above this = camera cut
SPIKE_KERNEL = 5        # median-filter kernel (frames) for single-frame spikes
SG_WINDOW = 11          # Savitzky-Golay window (frames), ~0.44 s
SG_POLY = 2


def _project_anchors(P: np.ndarray) -> Optional[np.ndarray]:
    """(4,2) image px of the anchors through P's z=0 plane, or None."""
    world = np.column_stack([ANCHORS[:, 0] - 52.5, ANCHORS[:, 1] - 34.0,
                             np.zeros(4), np.ones(4)])
    img = (P @ world.T).T
    if np.any(img[:, 2] <= 1e-6):
        return None
    return img[:, :2] / img[:, 2:3]


def _refit_P(anchor_px: np.ndarray) -> Optional[np.ndarray]:
    """Exact plane homography from the 4 anchor correspondences → 3x4 P."""
    src = (ANCHORS - [52.5, 34.0]).astype(np.float32)
    dst = anchor_px.astype(np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return None
    if not np.all(np.isfinite(H)):
        return None
    return np.column_stack([H[:, 0], H[:, 1], np.zeros(3), H[:, 2]])


def _smooth_segment(traj: np.ndarray) -> np.ndarray:
    """Smooth a (n, 8) anchor-coordinate trajectory in time."""
    n = len(traj)
    out = traj.copy()
    for c in range(traj.shape[1]):
        col = traj[:, c]
        if n >= SPIKE_KERNEL:
            col = medfilt(col, SPIKE_KERNEL)
        win = min(SG_WINDOW, n if n % 2 == 1 else n - 1)
        if win >= SG_POLY + 2:
            col = savgol_filter(col, win, SG_POLY)
        out[:, c] = col
    return out


def jitter_metric(frames: pd.DataFrame, mask: Optional[np.ndarray] = None) -> dict:
    """p50/p90/p99 of the 2nd temporal difference of anchor positions (px)."""
    P = frames[P_COLS].to_numpy().reshape(-1, 3, 4)
    has = frames["has_P"].to_numpy()
    pts = np.full((len(frames), 4, 2), np.nan)
    for i in np.where(has)[0]:
        a = _project_anchors(P[i])
        if a is not None:
            pts[i] = a
    acc = np.linalg.norm(np.diff(pts, n=2, axis=0), axis=2)
    ok = has[2:] & has[1:-1] & has[:-2]
    ok &= (np.diff(frames["frame"].to_numpy())[1:] == 1) \
        & (np.diff(frames["frame"].to_numpy())[:-1] == 1)
    if mask is not None:
        ok &= mask[2:]
    a = acc[ok]
    a = a[~np.isnan(a)]
    if not len(a):
        return {}
    return {"p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "n": int(len(a))}


def stabilize_projections(gs: GameState) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (frames_df, players_df, report) — stabilized copies."""
    frames = gs.frames.sort_values("frame").reset_index(drop=True).copy()
    P = frames[P_COLS].to_numpy().reshape(-1, 3, 4).copy()
    has = frames["has_P"].to_numpy()
    fnum = frames["frame"].to_numpy()

    # anchor trajectories
    pts = np.full((len(frames), 4, 2), np.nan)
    for i in np.where(has)[0]:
        a = _project_anchors(P[i])
        if a is not None:
            pts[i] = a
    valid = ~np.isnan(pts[:, 0, 0])

    # segments: consecutive valid frames without cuts
    segs = []
    start = None
    for i in range(len(frames)):
        if not valid[i]:
            if start is not None:
                segs.append((start, i))
                start = None
            continue
        if start is None:
            start = i
            continue
        step_break = fnum[i] != fnum[i - 1] + 1
        disp = np.nanmedian(np.linalg.norm(pts[i] - pts[i - 1], axis=1))
        if step_break or disp > CUT_PX:
            segs.append((start, i))
            start = i
    if start is not None:
        segs.append((start, len(frames)))

    # smooth + refit
    n_refit = 0
    for a, b in segs:
        if b - a < 3:
            continue
        traj = pts[a:b].reshape(b - a, 8)
        sm = _smooth_segment(traj).reshape(b - a, 4, 2)
        for k in range(b - a):
            newP = _refit_P(sm[k])
            if newP is not None:
                P[a + k] = newP
                n_refit += 1
    frames[P_COLS] = P.reshape(len(frames), 12)

    # recompute player pitch positions with the stabilized P
    players = gs.players.copy()
    P_by_frame = {int(fnum[i]): P[i] for i in range(len(frames)) if has[i]}
    px = np.full(len(players), np.nan)
    py = np.full(len(players), np.nan)
    pf = players["frame"].to_numpy()
    fx = players["foot_x_img"].to_numpy()
    fy = players["foot_y_img"].to_numpy()
    for i in range(len(players)):
        Pm = P_by_frame.get(int(pf[i]))
        if Pm is None:
            continue
        H = Pm[:, [0, 1, 3]]
        try:
            p = np.linalg.inv(H) @ np.array([fx[i], fy[i], 1.0])
        except np.linalg.LinAlgError:
            continue
        if abs(p[2]) < 1e-10:
            continue
        x, y = p[0] / p[2] + 52.5, p[1] / p[2] + 34.0
        if -10 <= x <= 115 and -10 <= y <= 78:
            px[i], py[i] = x, y
    players["pitch_x"] = px
    players["pitch_y"] = py

    report = {"segments": len(segs), "frames_refit": n_refit,
              "jitter_before": jitter_metric(gs.frames.sort_values("frame")
                                             .reset_index(drop=True)),
              "jitter_after": jitter_metric(frames)}
    return frames, players, report


def apply(gs: GameState, frames: pd.DataFrame, players: pd.DataFrame) -> None:
    _to_parquet(frames, gs.dir / "frames.parquet")
    _to_parquet(players, gs.dir / "players.parquet")


def main():
    ap = argparse.ArgumentParser(description="Stabilize stored homography sequence")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--half", type=int, default=None, choices=[1, 2],
                    help="period to stabilize (default: the only stored one)")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite frames/players parquet with stabilized values")
    args = ap.parse_args()

    gs = GameState.load(args.match, period=args.half)
    frames, players, rep = stabilize_projections(gs)
    print(f"segments: {rep['segments']} | frames refit: {rep['frames_refit']}")
    print(f"jitter before: {rep['jitter_before']}")
    print(f"jitter after:  {rep['jitter_after']}")
    if args.apply:
        apply(gs, frames, players)
        print("artifact rewritten (frames.parquet + players.parquet)")


if __name__ == "__main__":
    main()
