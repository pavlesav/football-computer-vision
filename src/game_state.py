"""
Persisted per-frame **game state** — the intermediate representation that sits
between perception (detection / tracking / team / homography) and analysis
(event detection, correction UI, validation).

Why this module exists
-----------------------
Until now every notebook re-ran the whole expensive perception pass (YOLO +
ResNet + PnLCalib) from scratch, so there was never a stable artifact to build
event logic on. `game_state.py` defines that artifact: one source of truth per
match, written once by :mod:`src.pipeline`, read by :mod:`src.events`, the
correction UI, and the validation harness.

This is the same "game state" that SoccerNet Game State Reconstruction is built
around: for every frame, the set of people on the pitch (image bbox + pitch XY +
team), the ball, and the camera projection.

On-disk layout
--------------
``output/game_state/{slug}/``:

  ``players.parquet`` — one row per (frame, track_id)::

      frame, track_id, team_id, conf,
      x1, y1, x2, y2,                 # image bbox (int)
      foot_x_img, foot_y_img,         # image px, bottom-centre of bbox
      pitch_x, pitch_y                # metres on 105x68, NaN if unavailable

  ``frames.parquet`` — one row per frame::

      frame, time_sec, period,
      is_gameplay, is_wide_shot,
      has_P, homog_source, homog_conf,
      P00..P23,                       # flattened 3x4 (row-major), NaN if no P
      ball_x1..ball_y2, ball_x_img, ball_y_img,
      ball_pitch_x, ball_pitch_y, ball_conf, ball_source

  ``ball.parquet`` — one row per raw ball candidate detection (a frame can
  have zero, one, or several)::

      frame, x1, y1, x2, y2, conf

  The frames table keeps only the best detection per frame (a convenient raw
  view); the candidates table is what :mod:`src.ball_tracker` consumes so the
  motion-consistent detection can be chosen over the top-confidence one.

  ``meta.json`` — slug, video, fps, period, frame range, counts, timestamp.

Coordinates: pitch is the FIFA-standard 105 x 68 m plane, origin at a corner
(x in 0..105 along the length, y in 0..68 across), matching
``run_pnlcalib_video.project_lines`` / ``run_demo.project_foot``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config

# 3x4 projection matrix flattened row-major into these column names.
P_COLS = [f"P{r}{c}" for r in range(3) for c in range(4)]


# ── In-memory row builders ───────────────────────────────────────────────────

@dataclass
class PlayerState:
    frame: int
    track_id: int
    bbox: tuple                       # (x1, y1, x2, y2) image px
    foot_xy_img: tuple                # (x, y) image px
    conf: float
    team_id: int = -1                 # 0/1/2; -1 = unassigned (filled post-pass)
    pitch_xy: Optional[tuple] = None  # (x, y) metres, or None

    def to_row(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        px, py = self.pitch_xy if self.pitch_xy is not None else (np.nan, np.nan)
        return {
            "frame": int(self.frame),
            "track_id": int(self.track_id),
            "team_id": int(self.team_id),
            "conf": float(self.conf),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "foot_x_img": float(self.foot_xy_img[0]),
            "foot_y_img": float(self.foot_xy_img[1]),
            "pitch_x": float(px),
            "pitch_y": float(py),
        }


@dataclass
class BallState:
    bbox: Optional[tuple] = None      # (x1, y1, x2, y2) image px
    xy_img: Optional[tuple] = None    # (x, y) image px
    pitch_xy: Optional[tuple] = None  # (x, y) metres
    conf: float = 0.0
    source: str = "none"              # detected | kalman | interp | predicted | none

    def to_cols(self) -> dict:
        bx1, by1, bx2, by2 = self.bbox if self.bbox is not None else (np.nan,) * 4
        ix, iy = self.xy_img if self.xy_img is not None else (np.nan, np.nan)
        px, py = self.pitch_xy if self.pitch_xy is not None else (np.nan, np.nan)
        return {
            "ball_x1": bx1, "ball_y1": by1, "ball_x2": bx2, "ball_y2": by2,
            "ball_x_img": float(ix) if ix == ix else np.nan,
            "ball_y_img": float(iy) if iy == iy else np.nan,
            "ball_pitch_x": float(px) if px == px else np.nan,
            "ball_pitch_y": float(py) if py == py else np.nan,
            "ball_conf": float(self.conf),
            "ball_source": self.source,
        }


@dataclass
class FrameState:
    frame: int
    time_sec: float
    period: int
    is_gameplay: bool
    is_wide_shot: bool
    P: Optional[np.ndarray] = None      # 3x4 or None
    homog_source: str = "none"
    homog_conf: float = 0.0
    players: list = field(default_factory=list)   # list[PlayerState]
    ball: Optional[BallState] = None

    def frame_row(self) -> dict:
        row = {
            "frame": int(self.frame),
            "time_sec": float(self.time_sec),
            "period": int(self.period),
            "is_gameplay": bool(self.is_gameplay),
            "is_wide_shot": bool(self.is_wide_shot),
            "has_P": self.P is not None,
            "homog_source": self.homog_source,
            "homog_conf": float(self.homog_conf),
        }
        Pflat = (np.asarray(self.P, dtype=float).reshape(-1)
                 if self.P is not None else [np.nan] * 12)
        row.update({c: float(v) for c, v in zip(P_COLS, Pflat)})
        row.update((self.ball or BallState()).to_cols())
        return row


# ── Writer ───────────────────────────────────────────────────────────────────

def game_state_dir(slug: str) -> Path:
    return Config.OUTPUT_GAME_STATE_DIR / slug


BALL_CAND_COLS = ["frame", "x1", "y1", "x2", "y2", "conf"]


def write_game_state(slug: str, frame_states: list, meta: dict,
                     ball_candidates: Optional[pd.DataFrame] = None) -> Path:
    """Expand a list of :class:`FrameState` into the Parquet tables + meta.

    ``ball_candidates`` — optional table of every raw ball detection
    (columns :data:`BALL_CAND_COLS`), consumed by :mod:`src.ball_tracker`.

    Returns the output directory. Overwrites any existing artifact for ``slug``.
    """
    out_dir = game_state_dir(slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    player_rows, frame_rows = [], []
    for fs in frame_states:
        frame_rows.append(fs.frame_row())
        for p in fs.players:
            player_rows.append(p.to_row())

    players_df = pd.DataFrame(player_rows) if player_rows else pd.DataFrame(
        columns=["frame", "track_id", "team_id", "conf", "x1", "y1", "x2", "y2",
                 "foot_x_img", "foot_y_img", "pitch_x", "pitch_y"])
    frames_df = pd.DataFrame(frame_rows)

    _to_parquet(players_df, out_dir / "players.parquet")
    _to_parquet(frames_df, out_dir / "frames.parquet")
    if ball_candidates is not None:
        _to_parquet(ball_candidates[BALL_CAND_COLS], out_dir / "ball.parquet")

    meta = {
        **meta,
        "slug": slug,
        "n_frames": len(frame_rows),
        "n_player_rows": len(player_rows),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write Parquet, falling back to pickle if pyarrow is unavailable."""
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        df.to_pickle(path.with_suffix(".pkl"))


def _read_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    pkl = path.with_suffix(".pkl")
    if pkl.exists():
        return pd.read_pickle(pkl)
    raise FileNotFoundError(path)


# ── Reader ───────────────────────────────────────────────────────────────────

class GameState:
    """Loaded game state for one match — thin convenience wrapper over the
    two DataFrames, used by :mod:`src.events`, validation, and the UI."""

    def __init__(self, slug: str, players: pd.DataFrame,
                 frames: pd.DataFrame, meta: dict):
        self.slug = slug
        self.players = players
        self.frames = frames.sort_values("frame").reset_index(drop=True)
        self.meta = meta

    @classmethod
    def load(cls, slug: str) -> "GameState":
        d = game_state_dir(slug)
        meta = json.loads((d / "meta.json").read_text())
        return cls(slug, _read_table(d / "players.parquet"),
                   _read_table(d / "frames.parquet"), meta)

    @property
    def fps(self) -> float:
        return float(self.meta.get("fps", 25.0))

    def get_P(self, frame: int) -> Optional[np.ndarray]:
        row = self.frames.loc[self.frames.frame == frame]
        if row.empty or not bool(row.iloc[0]["has_P"]):
            return None
        return row.iloc[0][P_COLS].to_numpy(dtype=float).reshape(3, 4)

    def players_at(self, frame: int) -> pd.DataFrame:
        return self.players.loc[self.players.frame == frame]

    def ball_trajectory(self) -> pd.DataFrame:
        """Per-frame ball pitch position (NaN where unavailable), frame-indexed."""
        cols = ["frame", "time_sec", "ball_pitch_x", "ball_pitch_y",
                "ball_x_img", "ball_y_img", "ball_source"]
        return self.frames[cols].copy()

    def ball_candidates(self) -> pd.DataFrame:
        """All raw ball candidate detections (``ball.parquet``); empty
        DataFrame with the right columns for legacy artifacts without it."""
        try:
            return _read_table(game_state_dir(self.slug) / "ball.parquet")
        except FileNotFoundError:
            return pd.DataFrame(columns=BALL_CAND_COLS)


# ── Homography trust ─────────────────────────────────────────────────────────
# The line-alignment confidence separates good from bad projections *within* a
# match, but its absolute scale is match-dependent (dusty bright paint on
# sut-mla saturates near 1.0; floodlit dec-mla paint has a green cast and even
# geometrically perfect projections score 0.2-0.5). A fixed threshold therefore
# does not transfer: 0.75 (QC-calibrated on sut-mla) rejected 100% of dec-mla.
# The gate anchors to the match's own score distribution instead, capped at the
# QC-validated absolute 0.75 and floored so a match whose stored projections are
# all garbage still trusts nothing.

HOMOG_CONF_ABS_MAX = 0.75   # QC-validated on sut-mla; never loosen above match evidence
HOMOG_CONF_ABS_MIN = 0.15   # never trust below this, whatever the distribution


def adaptive_conf_min(gs: "GameState") -> float:
    """Per-match trust threshold: 0.75 x the 75th percentile of the
    line-alignment confidence over fresh-PnLCalib wide frames."""
    f = gs.frames
    sel = f.loc[f["homog_source"].eq("pnlcalib") & f["is_wide_shot"], "homog_conf"]
    if len(sel) < 50:
        return HOMOG_CONF_ABS_MAX
    return float(np.clip(0.75 * sel.quantile(0.75),
                         HOMOG_CONF_ABS_MIN, HOMOG_CONF_ABS_MAX))


def trusted_frame_mask(gs: "GameState",
                       conf_min: Optional[float] = None) -> pd.Series:
    """Boolean per-frame mask: genuine wide shot with a confidently-aligned P.
    ``conf_min=None`` uses the per-match adaptive threshold."""
    if conf_min is None:
        conf_min = adaptive_conf_min(gs)
    f = gs.frames
    return f["has_P"] & f["is_wide_shot"] & (f["homog_conf"] >= conf_min)
