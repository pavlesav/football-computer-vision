"""
Kalman ball tracker over the persisted :mod:`src.game_state` (Milestone 2).

Why this exists
---------------
QC on `sut-mla` showed the event bottleneck is ball-state quality: YOLO loses
the ball on ~45% of frames (motion blur, occlusion, small size), with gaps of
12-64 frames during exactly the moments that matter — passes in flight. The old
2-frame velocity extrapolation (image space, heavily damped) and the events-side
8-frame linear interpolation could not bridge those, so several passes merged
into one multi-second "touch" and passes were under-counted.

Why pitch space, not image space
--------------------------------
The broadcast camera *pans to follow the ball*, so image-space constant-velocity
coasting is wrong during exactly the long gaps we need to bridge. In pitch
metres the ball's motion is camera-independent (a track even survives camera
cuts between wide angles), and the *ground track* of an airborne pass is
genuinely constant-velocity (horizontal velocity is unchanged in flight). The
projection of an airborne ball onto the pitch plane is biased while it is high,
but the ball is mostly detected near kick and reception (low), where the
projection is accurate — constant-velocity coasting between those is the right
physical model.

Design
------
Constant-velocity Kalman filter, state ``[x, y, vx, vy]`` in metres / m·s⁻¹:

* **Measurements** — ball detections projected to pitch XY with that frame's P,
  *only on trusted frames* (``is_wide_shot`` and ``homog_conf >= conf_min``,
  the same gate events use). Untrusted frames contribute no measurement.
* **Gating + selection** — candidates within a Mahalanobis gate; among gated
  candidates pick the motion-consistent one (min Mahalanobis, confidence as a
  tie-breaker) — NOT blindly the top-confidence detection.
* **Birth / confirm** — a track is born from a confident detection but only
  *reported* once confirmed by a second gated detection shortly after; a lone
  false positive (sock, bald head) never reaches the output.
* **Kicks** — an instantaneous velocity jump no smooth filter can chase. If
  confident detections are repeatedly rejected *and agree with each other*,
  the filter is wrong, not the detector: reinitialise from the detections
  (velocity from their finite difference).
* **Coasting** — no measurement ⇒ predict only; report as ``kalman`` up to
  `report_coast_max` frames, keep alive (unreported) a while longer for
  re-acquisition, then die. Coasting velocity decays slightly (friction).

Analysis-speed contract: this module is pure numpy/pandas over the persisted
artifact — **no video, no GPU** — so tracker parameters can be iterated in
seconds. Candidates come from ``ball.parquet`` (all detections per frame) when
present, else fall back to the best-detection columns in ``frames.parquet``.

Usage::

    from src.ball_tracker import track_ball
    ball = track_ball(GameState.load("sut-mla"))     # frame-indexed DataFrame

    python -m src.ball_tracker --match sut-mla       # coverage report
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .game_state import GameState, P_COLS, trusted_frame_mask, adaptive_conf_min

PITCH_L, PITCH_W = 105.0, 68.0
OUT_OF_PLAY_MARGIN_M = 8.0        # coasted this far outside the pitch ⇒ dead


@dataclass
class BallTrackerParams:
    fps: float = 25.0
    sigma_meas_m: float = 0.8      # measurement noise (homography jitter + bbox)
    # Tuned by masked-detection eval (hide k detections, measure reconstruction
    # error): 60 beats 25 by ~25% at k=10-20 on sut-mla+bud-sut with no gate
    # degradation (accepted counts stable, speeds stay physical). Long-gap p90
    # is parameter-invariant — bounded by unseen in-gap touches, not tuning.
    sigma_accel: float = 60.0      # process noise, white accel (m/s^2)
    gate_chi2: float = 13.82       # 99.9% chi-square, 2 dof
    gate_floor_m: float = 2.0      # always accept within this of the prediction
    birth_conf: float = 0.30       # detection conf needed to start a track
    confirm_window: int = 6        # frames after birth to find the 2nd hit
    conf_tiebreak: float = 2.0     # candidate cost = maha - this * conf
    reinit_count: int = 3          # consistent rejected confident dets ⇒ reinit
    reinit_window: int = 7         # ...within this many frames
    reinit_span_m: float = 8.0     # ...agreeing with each other this tightly
    report_coast_max: int = 30     # report 'kalman' while missed <= this (1.2s)
    kill_missed_max: int = 60      # drop the track entirely after this (2.4s)
    coast_decay: float = 0.99      # per-frame velocity decay while coasting
    init_speed_std: float = 15.0   # velocity std for a fresh single-detection track
    bridge_max_gap: int = 60       # hindsight-bridge gaps up to this (2.4s)
    # No real ball moves faster than a hard shot (~35 m/s). A bridge implying
    # more connects a false/off-pitch detection to a real one (full-half QC:
    # a 68m 'teleport' in 4 frames bridged at 428 m/s) — refuse it, and treat
    # reinit finite-difference velocities the same way.
    max_speed_ms: float = 40.0


@dataclass
class _Track:
    x: np.ndarray                  # state [x, y, vx, vy]
    P: np.ndarray                  # 4x4 covariance
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    born_frame: int = 0


def project_point_to_pitch(x_img: float, y_img: float,
                           P: np.ndarray) -> Optional[tuple]:
    """Image point → pitch metres (105x68, origin at a corner) via the z=0
    plane homography of a PnLCalib-convention 3x4 P. Mirrors
    ``run_demo.project_foot`` without importing the heavy demo module."""
    H = P[:, [0, 1, 3]]
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    p = Hinv @ np.array([x_img, y_img, 1.0])
    if abs(p[2]) < 1e-10:
        return None
    px = p[0] / p[2] + PITCH_L / 2
    py = p[1] / p[2] + PITCH_W / 2
    if -10 <= px <= PITCH_L + 10 and -10 <= py <= PITCH_W + 10:
        return (float(px), float(py))
    return None


# ── Measurement extraction ───────────────────────────────────────────────────

def _candidates_by_frame(gs: GameState,
                         conf_min: Optional[float]) -> dict[int, list]:
    """{frame: [(pitch_x, pitch_y, conf), ...]} on trusted frames only.
    Trust gate is shared with events (game_state.trusted_frame_mask) so the
    ball series never ingests a position events would not trust."""
    f = gs.frames
    trusted = trusted_frame_mask(gs, conf_min)
    P_by_frame = {
        int(r.frame): np.array([[getattr(r, c) for c in P_COLS[i*4:(i+1)*4]]
                                for i in range(3)])
        for r in f.loc[trusted].itertuples(index=False)
    }

    cand = gs.ball_candidates()
    if cand.empty:
        # Legacy artifact: fall back to the best-detection columns.
        det = f.loc[trusted & f["ball_source"].eq("detected"),
                    ["frame", "ball_x1", "ball_y1", "ball_x2", "ball_y2",
                     "ball_conf"]].rename(columns={
                         "ball_x1": "x1", "ball_y1": "y1",
                         "ball_x2": "x2", "ball_y2": "y2",
                         "ball_conf": "conf"})
        cand = det.dropna(subset=["x1"])

    out: dict[int, list] = {}
    for r in cand.itertuples(index=False):
        P = P_by_frame.get(int(r.frame))
        if P is None:
            continue
        # Bottom-centre of the bbox = ground contact point (as project_foot).
        xy = project_point_to_pitch((r.x1 + r.x2) / 2.0, float(r.y2), P)
        if xy is not None:
            out.setdefault(int(r.frame), []).append((xy[0], xy[1], float(r.conf)))
    return out


# ── The filter ───────────────────────────────────────────────────────────────

class BallKalman:
    """Single-object constant-velocity KF with birth/confirm/reinit/coast
    track management. Feed frames in order via :meth:`step`."""

    def __init__(self, params: BallTrackerParams):
        self.p = params
        dt = 1.0 / params.fps
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        q = params.sigma_accel ** 2
        self.Q = q * np.array(
            [[dt**4/4, 0,       dt**3/2, 0      ],
             [0,       dt**4/4, 0,       dt**3/2],
             [dt**3/2, 0,       dt**2,   0      ],
             [0,       dt**3/2, 0,       dt**2  ]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.R = (params.sigma_meas_m ** 2) * np.eye(2)
        self.track: Optional[_Track] = None
        self._rejected: list[tuple] = []   # (frame, x, y, conf) recent confident rejects

    # -- track lifecycle ------------------------------------------------------

    def _birth(self, frame: int, x: float, y: float,
               vx: float = 0.0, vy: float = 0.0, confirmed: bool = False):
        P0 = np.diag([self.p.sigma_meas_m**2 * 4, self.p.sigma_meas_m**2 * 4,
                      self.p.init_speed_std**2, self.p.init_speed_std**2])
        self.track = _Track(x=np.array([x, y, vx, vy]), P=P0,
                            born_frame=frame, confirmed=confirmed)
        self._rejected.clear()

    def reset(self):
        self.track = None
        self._rejected.clear()

    def _maybe_reinit(self, frame: int) -> Optional[float]:
        """Filter lost but the detector keeps agreeing with itself ⇒ trust it.
        Returns the conf of the detection reinitialised from, or None."""
        recent = [r for r in self._rejected if frame - r[0] < self.p.reinit_window]
        self._rejected = recent
        if len(recent) < self.p.reinit_count:
            return None
        xs = np.array([(r[1], r[2]) for r in recent])
        if np.ptp(xs[:, 0]) > self.p.reinit_span_m or \
           np.ptp(xs[:, 1]) > self.p.reinit_span_m:
            return None
        (f0, x0, y0, _), (f1, x1, y1, conf) = recent[0], recent[-1]
        dtf = max(f1 - f0, 1) / self.p.fps
        vx, vy = (x1 - x0) / dtf, (y1 - y0) / dtf
        if np.hypot(vx, vy) > self.p.max_speed_ms:
            vx = vy = 0.0          # detections too scattered to imply velocity
        self._birth(frame, x1, y1, vx, vy, confirmed=True)
        return conf

    # -- per-frame step -------------------------------------------------------

    def step(self, frame: int, cands: list[tuple]) -> dict:
        """Advance one frame. ``cands`` = [(pitch_x, pitch_y, conf), ...].
        Returns a row dict (source ∈ detected | kalman | none)."""
        t = self.track

        if t is not None:
            # Predict.
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q

            # Gate + select the motion-consistent candidate.
            S = self.H @ t.P @ self.H.T + self.R
            Sinv = np.linalg.inv(S)
            best, best_cost = None, np.inf
            for cx, cy, conf in cands:
                innov = np.array([cx - t.x[0], cy - t.x[1]])
                maha2 = float(innov @ Sinv @ innov)
                dist = float(np.hypot(*innov))
                if maha2 > self.p.gate_chi2 and dist > self.p.gate_floor_m:
                    if conf >= self.p.birth_conf:
                        self._rejected.append((frame, cx, cy, conf))
                    continue
                cost = maha2 - self.p.conf_tiebreak * conf
                if cost < best_cost:
                    best_cost, best = cost, (cx, cy, conf)

            if best is not None:
                z = np.array(best[:2])
                innov = z - self.H @ t.x
                K = t.P @ self.H.T @ np.linalg.inv(S)
                t.x = t.x + K @ innov
                t.P = (np.eye(4) - K @ self.H) @ t.P
                t.hits += 1
                t.missed = 0
                if not t.confirmed:
                    t.confirmed = True
                return self._row(frame, "detected", best[2])

            # No measurement accepted this frame.
            reinit_conf = self._maybe_reinit(frame)
            if reinit_conf is not None:
                return self._row(frame, "detected", reinit_conf)
            t.missed += 1
            t.x[2:] *= self.p.coast_decay
            unconfirmed_stale = (not t.confirmed
                                 and frame - t.born_frame > self.p.confirm_window)
            gone = (t.missed > self.p.kill_missed_max or unconfirmed_stale
                    or not self._in_bounds(t.x))
            if gone:
                self.reset()
            elif t.confirmed and t.missed <= self.p.report_coast_max:
                return self._row(frame, "kalman", 0.0)
            # fall through to birth logic below (track dead or unreportable)
            if self.track is not None:
                return {"frame": frame, "source": "none"}

        # No live track: try to birth from the most confident candidate.
        birthable = [c for c in cands if c[2] >= self.p.birth_conf]
        if birthable:
            cx, cy, conf = max(birthable, key=lambda c: c[2])
            self._birth(frame, cx, cy)
            return {"frame": frame, "source": "none"}   # unreported until confirmed
        return {"frame": frame, "source": "none"}

    def _in_bounds(self, x: np.ndarray) -> bool:
        m = OUT_OF_PLAY_MARGIN_M
        return (-m <= x[0] <= PITCH_L + m) and (-m <= x[1] <= PITCH_W + m)

    def _row(self, frame: int, source: str, conf: float) -> dict:
        t = self.track
        return {"frame": frame, "x": float(t.x[0]), "y": float(t.x[1]),
                "vx": float(t.x[2]), "vy": float(t.x[3]),
                "speed": float(np.hypot(t.x[2], t.x[3])),
                "source": source, "conf": float(conf),
                "missed": int(t.missed)}


# ── Hindsight bridging ───────────────────────────────────────────────────────

def _bridge_gaps(df: pd.DataFrame, gameplay_ok: np.ndarray,
                 max_gap: int, fps: float,
                 max_speed_ms: float = 40.0) -> pd.DataFrame:
    """Offline second pass: rewrite coasted/dead runs *bounded by detections*
    with a straight-line bridge.

    Why: QC showed the worst gap class is a kick followed immediately by a
    detection blackout — the causal filter coasts with pre-kick (near-zero)
    velocity and hovers at the kicker while the real ball is in flight. Once
    the ball is re-detected at the receiver we know, in hindsight, where it
    went; the ground track of a pass is linear, so interpolating between the
    two bounding detections is the physically right reconstruction. Bridges
    never cross non-gameplay frames (replays/cuts) and are labelled
    ``source='bridged'`` so nothing downstream mistakes them for evidence.
    """
    x = df["x"].to_numpy(copy=True)
    y = df["y"].to_numpy(copy=True)
    src = df["source"].to_numpy(dtype=object, copy=True)
    vx = df["vx"].to_numpy(copy=True)
    vy = df["vy"].to_numpy(copy=True)

    det_idx = np.where(src == "detected")[0]
    for a, b in zip(det_idx[:-1], det_idx[1:]):
        gap = b - a - 1
        if gap == 0 or gap > max_gap:
            continue
        if not gameplay_ok[a + 1:b].all():
            continue
        implied = np.hypot(x[b] - x[a], y[b] - y[a]) / (gap + 1) * fps
        if implied > max_speed_ms:
            continue   # endpoints can't be the same ball in continuous play
        t = np.arange(1, gap + 1) / (gap + 1)
        x[a + 1:b] = x[a] + (x[b] - x[a]) * t
        y[a + 1:b] = y[a] + (y[b] - y[a]) * t
        bvx = (x[b] - x[a]) / (gap + 1) * fps
        bvy = (y[b] - y[a]) / (gap + 1) * fps
        vx[a + 1:b], vy[a + 1:b] = bvx, bvy
        src[a + 1:b] = "bridged"

    out = df.copy()
    out["x"], out["y"], out["vx"], out["vy"] = x, y, vx, vy
    out["source"] = src
    out["speed"] = np.hypot(out["vx"], out["vy"])
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def track_ball(gs: GameState, conf_min: Optional[float] = None,
               params: Optional[BallTrackerParams] = None,
               bridge: bool = True) -> pd.DataFrame:
    """Run the Kalman tracker over a loaded game state.

    Returns one row per frame: ``frame, time_sec, x, y, vx, vy, speed, source,
    conf, missed`` — x/y in pitch metres, NaN where ``source == 'none'``.
    ``source`` ∈ detected | kalman (causal coast) | bridged (hindsight
    interpolation between bounding detections) | none.
    """
    params = params or BallTrackerParams(fps=gs.fps)
    cands = _candidates_by_frame(gs, conf_min)
    kf = BallKalman(params)

    frames = gs.frames.sort_values("frame")
    gameplay = dict(zip(frames["frame"].astype(int), frames["is_gameplay"]))
    tsec = dict(zip(frames["frame"].astype(int), frames["time_sec"]))

    rows = []
    for f in frames["frame"].astype(int):
        if not gameplay.get(f, True):
            # Replay / graphics: whatever happens on screen is not live play.
            kf.reset()
            rows.append({"frame": f, "source": "none"})
        else:
            rows.append(kf.step(f, cands.get(f, [])))
        rows[-1]["time_sec"] = tsec.get(f, np.nan)

    df = pd.DataFrame(rows)
    for c in ("x", "y", "vx", "vy", "speed", "conf", "missed"):
        if c not in df:
            df[c] = np.nan
    df = df[["frame", "time_sec", "x", "y", "vx", "vy", "speed",
             "source", "conf", "missed"]]
    if bridge:
        gameplay_ok = frames["is_gameplay"].to_numpy(dtype=bool)
        df = _bridge_gaps(df, gameplay_ok, params.bridge_max_gap, params.fps,
                          params.max_speed_ms)
    return df


def coverage_report(gs: GameState, ball: pd.DataFrame) -> str:
    n = len(ball)
    src = ball["source"].value_counts().to_dict()
    raw_det = int(gs.frames["ball_source"].eq("detected").sum())
    have = ball["source"].isin(["detected", "kalman", "bridged"])
    lines = [
        f"frames: {n}",
        f"raw YOLO best-detection frames: {raw_det} ({100*raw_det/max(n,1):.1f}%)",
        f"tracker output: {src}",
        f"ball position available: {int(have.sum())} ({100*have.mean():.1f}%)",
        f"speed p50/p95/max (m/s): "
        f"{ball.speed.quantile(0.5):.1f} / {ball.speed.quantile(0.95):.1f} / "
        f"{ball.speed.max():.1f}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Kalman ball tracking over a game state")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--half", type=int, default=None, choices=[1, 2],
                    help="period to track (default: the only stored one)")
    ap.add_argument("--conf_min", type=float, default=None,
                    help="Trust threshold; default = per-match adaptive")
    args = ap.parse_args()

    gs = GameState.load(args.match, period=args.half)
    ball = track_ball(gs, conf_min=args.conf_min)
    print(coverage_report(gs, ball))


if __name__ == "__main__":
    main()
