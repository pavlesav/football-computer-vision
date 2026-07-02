"""
Demo video for FSCG presentation.

Pipeline per frame: YOLO + BoT-SORT for player/ball detection, smoothed
manual-seed homography (data/manual_calibration/), manual ball annotations
(data/ball_annotations/), and a per-player speed accumulator. The notebook
``notebooks/04_demo_video.ipynb`` is the iteration sandbox; this module is
the standalone CLI runner with the same logic.

Step 1 — find track IDs in the clip:
    python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16 --discover

Step 2 — edit PLAYER_NAMES below, then render final demo:
    python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16
"""
import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.camera_motion import CameraMotionTracker
from src.manual_calibration import load_match_seeds
from src.ball_annotation import (
    load_ball_annotations,
    interpolate_ball as interpolate_ball_annotation,
)
from src.run_pnlcalib_video import (
    load_models,
    predict_one_frame,
    project_lines,
    ProjectionSmoother,
    refine_projection_to_lines,
    check_projection_sanity,
    check_line_alignment,
)

# ── Edit this after running --discover ────────────────────────────────────────
# track_id: ("Display Name", team_id, jersey_no)
# team_id: 0 = Sutjeska (blue), 1 = Mladost (yellow), 2 = Other (ref/staff)
PLAYER_NAMES: dict[int, tuple[str, int, int | None]] = {
    # ── SUTJESKA ─────────────────────────────────────────────────────────────
    289: ('A.Babic',       0,  72),  513: ('A.Babic',       0,  72),
    319: ('B. Kopitovic',  0,  15),  519: ('B. Kopitovic',  0,  15),
    106: ('A. Golubovic',  0,   2),  611: ('A. Golubovic',  0,   2),
    339: ('M. Vracar',     0,   4),  653: ('M. Vracar',     0,   4),
    397: ('D. Hocko',      0,  44),
    382: ('J. Cadjenovic', 0,  20),
    596: ('V. Kalezic',    0,  70),  419: ('V. Kalezic',    0,  70),
    415: ('I. Vukcevic',   0,   7),  656: ('I. Vukcevic',   0,   7),
    535: ('M. Jukovic',    0,  88),  395: ('M. Jukovic',    0,  88),
    428: ('V. Cavor',      0,  24),  621: ('V. Cavor',      0,  24),
    # ── MLADOST ──────────────────────────────────────────────────────────────
    675: ('B.Radanovic',   1,  31),  716: ('B.Radanovic',   1,  31),
    439: ('M.Badnjar',     1,   3),  651: ('M.Badnjar',     1,   3),
    638: ('Z.Ceklic',      1,   6),  437: ('Z.Ceklic',      1,   6),
    429: ('J.Vujisic',     1,   5),  605: ('J.Vujisic',     1,   5),
    412: ('V.Vickovic',    1,  18),  544: ('V.Vickovic',    1,  18),
    394: ('D.Kontic',      1,   8),
    392: ('N.Radusinovic', 1,  16),
    389: ('D.Vukovic',     1,  77),
    266: ('J.Roganovic',   1,  29),
    253: ('N.A.Cordoba',   1,  28),
    230: ('L.Knezevic',    1,   7),  604: ('L.Knezevic',    1,   7),
    # ── REFEREES ─────────────────────────────────────────────────────────────
    337: ('Sudija', 2, None),  393: ('Sudija', 2, None),
    502: ('Sudija', 2, None),  494: ('Sudija', 2, None),
}

# When two players' paths cross, BoT-SORT can swap their IDs.
# Add an entry for each swap: (seconds_into_clip, {old_id: new_id, new_id: old_id})
TRACK_REMAPS: list[tuple[float, dict[int, int]]] = [
    (14, {253: 339, 653: 253}),
]


def resolve_tid(tid: int, render_sec: float) -> int:
    """Translate a raw tracker ID to its corrected ID at the given clip time."""
    for threshold, remap in sorted(TRACK_REMAPS, key=lambda x: x[0]):
        if render_sec >= threshold:
            tid = remap.get(tid, tid)
    return tid


PREROLL_SEC = 15            # seconds before start_ts: stabilises tracking + seeds PnLCalib
MINIMAP_W = 340             # minimap width in pixels
MINIMAP_H = int(MINIMAP_W * 68 / 105)
SPEED_EMA = 0.25            # lower = smoother speed readings
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 2

# Ball detection is sparse — YOLO loses the ball on dribbles, in the grass,
# and behind partial occlusions. The render path consults sources in this
# order: manual annotation (data/ball_annotations/, linearly interpolated),
# YOLO this frame, then velocity-extrapolated YOLO from the last two
# detections with damping that fades the prediction as the gap grows.
BALL_CONF_THRESHOLD = 0.08
BALL_MAX_GAP_FRAMES = 30        # ~1.2s @ 25fps


# ── Speed / distance tracker ──────────────────────────────────────────────────

class PlayerStats:
    """Per-player speed (km/h) and cumulative distance (m).

    Pitch positions arrive noisy: bbox jitter, stride oscillation (foot
    swinging fwd/back through the gait cycle), and small frame-to-frame
    homography drift each contribute 0.05–0.50m of fake motion at 25 fps.
    The class smooths in two stages — an EMA on the raw pitch position,
    then a window-end-to-end speed read off the smoothed track — so two
    side-by-side runners with out-of-phase strides report similar values.

    A few guards keep the readout sane:
      * MAX_SPEED_KMH caps at a realistic sprint ceiling. Lower than the
        Bolt-style 40 km/h on purpose so single-frame outliers don't pin
        the EMA to the cap for seconds.
      * TELEPORT_M catches ID swaps and bad projections — anything moving
        more than 3m in one frame is treated as data corruption: snap,
        decay speed, drop the window.
      * NOISE_FLOOR_M absorbs sub-pixel jitter so standing players don't
        rack up distance.
      * SPEED_WARMUP_FRAMES suppresses the speed reading on freshly-born
        tracks. The first few frames after a track is assigned only have
        one or two history points; a single bbox glitch then dominates
        the end-to-end and routinely reads "40 km/h" for a stationary
        player.
    """
    MAX_SPEED_KMH = 36.0
    TELEPORT_M = 3.0
    NOISE_FLOOR_M = 0.12
    POSITION_EMA = 0.18
    SPEED_WINDOW_FRAMES = 19    # ~760ms at 25 fps — covers two strides
    SPEED_WARMUP_FRAMES = 7

    def __init__(self, fps: float):
        self.fps = fps
        self._smoothed: dict[int, np.ndarray] = {}
        self._history: dict[int, deque] = {}
        self._speed: dict[int, float] = {}
        self._dist: dict[int, float] = {}

    def update(self, track_id: int, pitch_xy: np.ndarray,
               canonical_id=None) -> None:
        """Update positional state for ``track_id`` and accumulate distance.

        ``canonical_id`` (typically the display name) is the key used for
        distance accumulation, so the same physical player keeps adding to
        their cumulative distance even when BoT-SORT assigns them a fresh
        track_id after they leave and re-enter the frame. Speed and the
        position window remain per track_id — they should reset on a new
        track because there's no continuous motion across the gap.
        """
        if canonical_id is None:
            canonical_id = track_id

        prev = self._smoothed.get(track_id)
        if prev is None:
            # First sample for this track — seed everything from the raw point.
            self._smoothed[track_id] = pitch_xy.copy()
            self._history[track_id] = deque([pitch_xy.copy()])
            return

        # Teleport check is on the RAW per-frame delta. A 5m glitch otherwise
        # only moves the EMA-smoothed position by 1.5m and slips below
        # TELEPORT_M, leaking into accumulated distance.
        d_raw = float(np.linalg.norm(pitch_xy - prev))
        if d_raw > self.TELEPORT_M:
            self._smoothed[track_id] = pitch_xy.copy()
            self._history[track_id] = deque([pitch_xy.copy()])
            self._speed[track_id] = (1 - SPEED_EMA) * self._speed.get(track_id, 0.0)
            return

        smoothed = self.POSITION_EMA * pitch_xy + (1 - self.POSITION_EMA) * prev
        d = float(np.linalg.norm(smoothed - prev))
        if d > self.NOISE_FLOOR_M:
            self._dist[canonical_id] = self._dist.get(canonical_id, 0.0) + d

        self._smoothed[track_id] = smoothed
        history = self._history[track_id]
        history.append(smoothed)
        if len(history) > self.SPEED_WINDOW_FRAMES:
            history.popleft()

        # Window-based speed: end-to-end displacement of smoothed positions
        # over the window divided by elapsed time. Suppressed until the
        # window has SPEED_WARMUP_FRAMES entries — early-frame readings
        # are dominated by the first sample's bbox jitter and routinely
        # report fake sprint speeds.
        if len(history) >= self.SPEED_WARMUP_FRAMES:
            d_window = float(np.linalg.norm(history[-1] - history[0]))
            elapsed_sec = (len(history) - 1) / self.fps
            speed_kmh = min(d_window / elapsed_sec * 3.6, self.MAX_SPEED_KMH)
            prev_speed = self._speed.get(track_id, speed_kmh)
            self._speed[track_id] = SPEED_EMA * speed_kmh + (1 - SPEED_EMA) * prev_speed
        else:
            self._speed[track_id] = (1 - SPEED_EMA) * self._speed.get(track_id, 0.0)

    def drop(self, track_id: int) -> None:
        """Forget a track's history so the next update starts fresh."""
        self._smoothed.pop(track_id, None)
        self._history.pop(track_id, None)

    def speed(self, track_id: int) -> float:
        return self._speed.get(track_id, 0.0)

    def dist(self, canonical_id) -> float:
        """Return cumulative metres travelled for ``canonical_id`` (typically
        the display name) — sums across every track_id assigned to the same
        physical player throughout the clip."""
        return self._dist.get(canonical_id, 0.0)


# ── Wide-angle gate ───────────────────────────────────────────────────────────

def is_wide_shot(players: list, frame_h: int,
                 min_players: int = 7, max_tallest_frac: float = 0.22) -> bool:
    """Is this a wide tactical shot (homography meaningful) or a close-up (not)?

    Cheap — uses only the already-detected player boxes, no extra YOLO pass.
    Close-ups have few players and the biggest box fills a big chunk of the frame.

    Judged on the 3rd-tallest box, not the tallest: on grounds with a low
    camera (dec-mla) a genuinely wide shot often has one player near the lens
    whose box breaches the height cap, and the tallest-box rule mislabelled
    532/6000 such frames as close-ups (QC-verified all genuinely wide). In a
    real close-up *all* boxes are big, so the 3rd-tallest still catches it.
    """
    if len(players) < min_players:
        return False
    heights = sorted((p['bbox'][3] - p['bbox'][1]) / frame_h for p in players)
    return heights[-3] <= max_tallest_frac


# ── Projection helper ─────────────────────────────────────────────────────────

def project_foot(bbox, P: np.ndarray) -> Optional[np.ndarray]:
    """Project foot point (bottom-centre of bbox) to pitch XY in metres."""
    x1, y1, x2, y2 = bbox
    foot = np.array([(x1 + x2) / 2.0, float(y2), 1.0])
    H = np.linalg.inv(P[:, [0, 1, 3]])
    p = H @ foot
    if abs(p[2]) < 1e-10:
        return None
    px = p[0] / p[2] + 52.5
    py = p[1] / p[2] + 34.0
    if -10 <= px <= 115 and -10 <= py <= 78:
        return np.array([px, py])
    return None


def compute_viewport_polygon(
    P: np.ndarray, frame_w: int, frame_h: int,
) -> Optional[list[tuple[float, float]]]:
    """4-corner polygon (in pitch metres, natural coords) of the camera frame.

    Projects the four image corners onto the z=0 pitch plane and returns
    them in clockwise order. Returns None if any corner ends up at or
    behind the camera plane (degenerate). Out-of-pitch points are returned
    as-is — the minimap render clips them when drawing.
    """
    try:
        H = np.linalg.inv(P[:, [0, 1, 3]])
    except np.linalg.LinAlgError:
        return None
    pts: list[tuple[float, float]] = []
    for ix, iy in [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]:
        h = H @ np.array([ix, iy, 1.0])
        if abs(h[2]) < 1e-6:
            return None
        pts.append((h[0] / h[2] + 52.5, h[1] / h[2] + 34.0))
    return pts


# ── Minimap ───────────────────────────────────────────────────────────────────

def _mm_pt(x: float, y: float, pad: int) -> tuple[int, int]:
    return (int(x / 105 * MINIMAP_W) + pad, int(y / 68 * MINIMAP_H) + pad)


def draw_minimap(
    frame: np.ndarray,
    player_pts: dict[int, tuple[np.ndarray, int]],
    ball_pt: Optional[np.ndarray],
    viewport_pts: Optional[list[tuple[float, float]]] = None,
) -> None:
    pad = 8
    cw = MINIMAP_W + 2 * pad
    ch = MINIMAP_H + 2 * pad
    mm = np.full((ch, cw, 3), (18, 55, 18), dtype=np.uint8)

    def _ln(a, b, c=(170, 170, 170)):
        cv2.line(mm, _mm_pt(*a, pad), _mm_pt(*b, pad), c, 1)

    def _rc(tl, br, c=(170, 170, 170)):
        cv2.rectangle(mm, _mm_pt(*tl, pad), _mm_pt(*br, pad), c, 1)

    # Pitch outline
    _rc((0, 0), (105, 68))
    # Halfway line + centre circle
    _ln((52.5, 0), (52.5, 68))
    r = max(1, int(9.15 / 105 * MINIMAP_W))
    cv2.circle(mm, _mm_pt(52.5, 34, pad), r, (170, 170, 170), 1)
    # Penalty areas
    _rc((0, 13.84), (16.5, 54.16))
    _rc((88.5, 13.84), (105, 54.16))
    # Goal areas
    _rc((0, 24.84), (5.5, 43.16))
    _rc((99.5, 24.84), (105, 43.16))
    # Penalty spots
    for sx, sy in [(11, 34), (94, 34)]:
        cv2.circle(mm, _mm_pt(sx, sy, pad), 2, (170, 170, 170), -1)

    # Camera viewport: translucent yellow polygon over the area the camera
    # can currently see, drawn between pitch lines and player dots so player
    # / ball circles read on top of it.
    if viewport_pts is not None:
        poly = np.array([_mm_pt(x, y, pad) for x, y in viewport_pts],
                        dtype=np.int32)
        overlay = mm.copy()
        cv2.fillPoly(overlay, [poly], (80, 230, 255))   # warm yellow (BGR)
        cv2.addWeighted(overlay, 0.35, mm, 0.65, 0, dst=mm)
        cv2.polylines(mm, [poly], True, (60, 240, 255), 2, cv2.LINE_AA)

    for tid, (xy, team_id) in player_pts.items():
        color = Config.TEAM_COLORS.get(team_id, (180, 180, 180))
        mx, my = _mm_pt(float(xy[0]), float(xy[1]), pad)
        if 0 <= mx < cw and 0 <= my < ch:
            cv2.circle(mm, (mx, my), 5, color, -1)
            cv2.circle(mm, (mx, my), 5, (0, 0, 0), 1)

    if ball_pt is not None:
        bx, by = _mm_pt(float(ball_pt[0]), float(ball_pt[1]), pad)
        if 0 <= bx < cw and 0 <= by < ch:
            cv2.circle(mm, (bx, by), 4, (0, 220, 80), -1)
            cv2.circle(mm, (bx, by), 4, (0, 0, 0), 1)

    fh, fw = frame.shape[:2]
    x1 = (fw - cw) // 2
    y1 = fh - ch - 14
    roi = frame[y1:y1 + ch, x1:x1 + cw]
    frame[y1:y1 + ch, x1:x1 + cw] = cv2.addWeighted(roi, 0.2, mm, 0.8, 0)
    cv2.rectangle(frame, (x1, y1), (x1 + cw, y1 + ch), (160, 160, 160), 1)


# ── Per-player drawing ────────────────────────────────────────────────────────

def draw_ellipse(frame: np.ndarray, bbox, color) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    ax = max(4, (x2 - x1) // 2)
    ay = max(3, (x2 - x1) // 5)
    cv2.ellipse(frame, (cx, y2), (ax, ay), 0, -45, 235, color, 2, cv2.LINE_AA)


def draw_ball_triangle(frame: np.ndarray, bbox) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    s = 12
    tri = np.array([[cx, y1 - s - 5], [cx - s, y1 - 5], [cx + s, y1 - 5]], np.int32)
    cv2.fillPoly(frame, [tri], (0, 220, 80))
    cv2.polylines(frame, [tri], True, (0, 0, 0), 1)


_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        for path in [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/verdanab.ttf",
        ]:
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                pass
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _bgr_to_rgb(c) -> tuple:
    return (int(c[2]), int(c[1]), int(c[0]))


def _text_badge(frame: np.ndarray, text: str, cx: int, cy: int,
                bg_color, text_color=(0, 0, 0), font_size: int = 17) -> None:
    font = _load_font(font_size)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    lx, ty = cx - tw // 2, cy - th
    pad = 5
    try:
        draw.rounded_rectangle([lx - pad, ty - pad, lx + tw + pad, cy + pad],
                                radius=4, fill=_bgr_to_rgb(bg_color))
    except AttributeError:
        draw.rectangle([lx - pad, ty - pad, lx + tw + pad, cy + pad],
                       fill=_bgr_to_rgb(bg_color))
    draw.text((lx, ty), text, font=font, fill=_bgr_to_rgb(text_color))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _stroke_text(frame, text, pos, scale, color, thickness=2) -> None:
    """White text with a black stroke — readable against any pitch background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


# team_id used for referees / staff. Keep stat readouts off these so the
# overlay is reserved for actual players.
REF_TEAM_ID = 2


def draw_player_annotation(frame, bbox, name, team_id, speed_kmh, dist_m, track_id, jersey_no=None) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    color = Config.TEAM_COLORS.get(team_id, (180, 180, 180))
    draw_ellipse(frame, bbox, color)

    # Unlabeled players get ellipse only — no badge, no stats.
    if not name:
        return

    label = f"{name},{jersey_no}" if jersey_no is not None else name
    _text_badge(frame, label, cx, y1 - 6, color)

    # Referees / staff: name above (already drawn), no speed / distance.
    if team_id == REF_TEAM_ID:
        return

    speed_scale = 0.6

    if speed_kmh >= 0.5:
        spd = f"{speed_kmh:.1f} km/h"
        (sw, sh), _ = cv2.getTextSize(spd, cv2.FONT_HERSHEY_SIMPLEX, speed_scale, 2)
        _stroke_text(frame, spd, (cx - sw // 2, y2 + sh + 6),
                     speed_scale, (255, 255, 255), thickness=2)
    # dist_m is intentionally unused — the cumulative distance readout was
    # removed for the demo. The accumulator still runs in PlayerStats so
    # we can switch it back on later if needed.


REF_COLOR = (0, 165, 255)  # orange — referee / staff


def draw_discover_box(frame, bbox, track_id, name=None, team_id=2, jersey_no=None) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    color = Config.TEAM_COLORS.get(team_id, REF_COLOR) if name is not None else (0, 255, 0)
    draw_ellipse(frame, bbox, color)
    if name is not None:
        label = f"#{jersey_no}, {name}" if jersey_no is not None else name
    else:
        label = str(track_id) if track_id >= 0 else "?"
    _text_badge(frame, label, cx, y1 - 6, color)


def draw_timestamp(frame: np.ndarray, abs_frame: int, fps: float) -> None:
    total_s = int(abs_frame / fps)
    h, m, s = total_s // 3600, (total_s % 3600) // 60, total_s % 60
    text = f"{h:02d}:{m:02d}:{s:02d}"
    font = _load_font(18)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = 6
    draw.rounded_rectangle([10 - pad, 10 - pad, 10 + tw + pad, 10 + th + pad],
                            radius=4, fill=(20, 20, 20))
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── Smoothed manual-seed trajectory ───────────────────────────────────────────

def build_smoothed_seed_track(
    manual_seeds: dict[int, np.ndarray],
    sigma_frames: float = 4.0,
) -> dict[int, np.ndarray]:
    """Pre-compute a temporally-smoothed manual-seed P trajectory.

    Why this exists: linear-interpolating between adjacent manual seeds
    produces a piecewise-linear track with a corner at every seeded frame.
    Combined with per-frame jitter from PnLCalib + flow that gets blended
    in, the rendered overlay looks like it "snaps" to a correct pose at
    each seed and then drifts away until the next seed yanks it back —
    which is exactly the artefact the user reported.

    The fix: build a dense per-frame P track by linear interpolation of
    seeds, then run a Gaussian filter over time across the 12 entries of
    each 3×4 P matrix. The filter rounds the corners — so frames between
    seeds inherit motion that's continuous in both position AND velocity,
    instead of jumping when one seed pair takes over from another. The
    pipeline is no longer in the loop within seed range, so PnLCalib /
    flow jitter cannot leak through.

    sigma_frames: Gaussian kernel σ in frames. 4 = ~160 ms at 25 fps,
    which rounds the every-10-frames corners without flattening genuine
    camera pans implied by widely-spaced seeds. Increase for more
    smoothness (and more lag relative to the GT poses); decrease if the
    overlay wobbles within a single seed gap.

    Returns a dict {abs_frame: P (3×4)} covering every frame between the
    earliest and latest seeded frame. Frames outside that range fall back
    to the pipeline at render time.
    """
    if not manual_seeds:
        return {}
    sorted_frames = sorted(manual_seeds.keys())
    first, last = sorted_frames[0], sorted_frames[-1]
    n_frames = last - first + 1
    if n_frames < 2:
        return {first: manual_seeds[first].copy()}

    # Dense linear interpolation between consecutive seeds.
    track = np.empty((n_frames, 3, 4), dtype=np.float64)
    after_idx = 1
    for offset in range(n_frames):
        abs_f = first + offset
        while (after_idx < len(sorted_frames)
                and sorted_frames[after_idx] < abs_f):
            after_idx += 1
        if abs_f in manual_seeds:
            track[offset] = manual_seeds[abs_f]
            continue
        f1 = sorted_frames[after_idx - 1]
        f2 = sorted_frames[after_idx]
        t = (abs_f - f1) / (f2 - f1)
        track[offset] = (1 - t) * manual_seeds[f1] + t * manual_seeds[f2]

    # Gaussian smoothing along time. Try scipy first; fall back to a small
    # NumPy convolution so the dependency stays optional.
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(track, sigma=sigma_frames,
                                       axis=0, mode="nearest")
    except ImportError:
        radius = int(np.ceil(3 * sigma_frames))
        kx = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (kx / sigma_frames) ** 2)
        kernel /= kernel.sum()
        # Reflect-pad along time, convolve per (i, j) entry.
        padded = np.pad(track, ((radius, radius), (0, 0), (0, 0)),
                        mode="edge")
        smoothed = np.empty_like(track)
        for offset in range(n_frames):
            window = padded[offset:offset + 2 * radius + 1]
            smoothed[offset] = (window * kernel[:, None, None]).sum(axis=0)

    return {first + i: smoothed[i] for i in range(n_frames)}


# ── Timestamp parser ──────────────────────────────────────────────────────────

def parse_ts(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="sut-mla")
    parser.add_argument("--start_ts", default="1:04:48",
                        help="Start timestamp in the video file (HH:MM:SS or MM:SS)")
    parser.add_argument("--duration_sec", type=int, default=16)
    parser.add_argument("--discover", action="store_true",
                        help="Output track IDs only — no PnLCalib, no names")
    parser.add_argument("--preroll_sec", type=int, default=PREROLL_SEC,
                        help="Seconds before start_ts to run tracking silently")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="PnLCalib EMA smoothing (lower = smoother)")
    parser.add_argument("--seed_sigma", type=float, default=6.0,
                        help="Gaussian σ (frames) for the manual-seed "
                              "trajectory. Higher = smoother overlay AND wider "
                              "support across sparse-seed gaps (so neighbour "
                              "seeds dominate inside the gap), but more lag "
                              "relative to labeled GT poses. 4–6 for densely-"
                              "seeded clips, 8+ for clips with gaps >15 frames.")
    args = parser.parse_args()

    video_path = str(Config.MATCH_VIDEOS[args.match])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_sec = parse_ts(args.start_ts)
    start_abs = int(start_sec * fps)
    preroll_frames = int(args.preroll_sec * fps)
    render_frames = int(args.duration_sec * fps)
    total_frames = preroll_frames + render_frames
    clip_start = max(0, start_abs - preroll_frames)

    mode = "DISCOVER" if args.discover else "DEMO"
    print(f"Match: {args.match} | {fw}x{fh_v} @ {fps}fps")
    print(f"Target clip: {args.start_ts} (frame {start_abs}) → +{args.duration_sec}s")
    print(f"Pre-roll: {args.preroll_sec}s from frame {clip_start}")
    print(f"Mode: {mode}")

    # ── Models ────────────────────────────────────────────────────────
    print("Loading YOLO...")
    yolo = YOLO(Config.resolve_yolo_model())

    pnl_models = None
    if not args.discover:
        print("Loading PnLCalib...")
        pnl_models = load_models(args.device)

    # ── Output ────────────────────────────────────────────────────────
    demo_dir = Config.OUTPUT_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = args.start_ts.replace(":", "-")
    out_path = str(demo_dir / f"{args.match}_{ts_safe}_{mode.lower()}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh_v))

    # ── State ─────────────────────────────────────────────────────────
    smoother = ProjectionSmoother(alpha=args.alpha, max_stale_frames=int(fps * 3))
    motion_tracker = CameraMotionTracker() if pnl_models else None
    player_stats = PlayerStats(fps)
    manual_seeds = load_match_seeds(args.match) if pnl_models else {}
    smoothed_seed_track = build_smoothed_seed_track(
        manual_seeds, sigma_frames=args.seed_sigma,
    )
    ball_annotations = load_ball_annotations(args.match) if pnl_models else {}
    if ball_annotations:
        bf = sorted(ball_annotations.keys())
        print(f"Loaded {len(ball_annotations)} ball annotation(s) for "
              f"{args.match} (range {bf[0]}–{bf[-1]})")
    if manual_seeds:
        sorted_seed_frames = sorted(manual_seeds.keys())
        print(f"Loaded {len(manual_seeds)} manual seed(s) for {args.match} "
              f"(range {sorted_seed_frames[0]}–{sorted_seed_frames[-1]}, "
              f"σ={args.seed_sigma:.1f} frames)")
    homog_source_counts: dict[str, int] = {}

    # Reset BoT-SORT's global ID counter so track IDs in PLAYER_NAMES match
    # this run. Otherwise CLI runs after notebook iteration get fresh IDs and
    # all the name mappings miss.
    if hasattr(yolo, "predictor") and yolo.predictor is not None:
        yolo.predictor = None
    from ultralytics.trackers.basetrack import BaseTrack
    BaseTrack._count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    P_smooth = None
    player_pitch_pts: dict[int, tuple[np.ndarray, int]] = {}
    ball_pitch_pt: Optional[np.ndarray] = None

    # Ball-tracking state for the velocity-extrapolation logic — see the
    # BALL_* constants at the top of the file. Last 2 detected bboxes,
    # their absolute frame indices, and the corresponding pitch-XY (when
    # homography is available).
    _ball_bboxes: list[list[int]] = []
    _ball_frames: list[int] = []
    _ball_pitch: list[Optional[np.ndarray]] = []

    for frame_i in tqdm(range(total_frames), desc=mode):
        ret, frame = cap.read()
        if not ret:
            break

        is_render = frame_i >= preroll_frames
        abs_frame = clip_start + frame_i
        # Hoisted up so stats accumulation inside this frame can resolve the
        # display tid (and therefore the canonical player name) the same way
        # the renderer does. Negative during pre-roll — resolve_tid is a
        # no-op there since no remap threshold has fired.
        render_sec = (frame_i - preroll_frames) / fps

        # ── Detection + tracking ──────────────────────────────────────
        # Run inference at the lowest of the two thresholds so weak ball
        # detections still survive the YOLO filter; the per-class checks
        # below sort them out.
        results = yolo.track(
            frame,
            conf=min(Config.PLAYER_CONF_THRESHOLD, BALL_CONF_THRESHOLD),
            tracker=f"{Config.TRACKER_TYPE}.yaml",
            persist=True,
            verbose=False,
        )[0]
        boxes = results.boxes

        players = []
        best_ball_bbox = None
        best_ball_conf = -1.0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i])
            tid = int(boxes.id[i]) if boxes.id is not None else -1
            if cls_id == 0 and conf >= Config.PLAYER_CONF_THRESHOLD:
                players.append({"bbox": bbox, "conf": conf, "track_id": tid})
            elif cls_id == 1 and conf >= BALL_CONF_THRESHOLD and conf > best_ball_conf:
                best_ball_bbox, best_ball_conf = bbox, conf

        # ── Homography ────────────────────────────────────────────────
        # Within the manual-seed range, the smoothed seed trajectory IS
        # the answer. We don't run PnLCalib / flow / refine / smoother at
        # all — those are the sources of the snap-and-drift artefact.
        # Outside the seed range we fall back to the full pipeline.
        if pnl_models is not None:
            seed_P = smoothed_seed_track.get(abs_frame)
            if seed_P is not None:
                P_smooth = seed_P
                source = "smooth_seed"
            else:
                P_raw, _ = predict_one_frame(
                    frame, pnl_models, fw, fh_v,
                    manual_seeds=manual_seeds, abs_frame=abs_frame,
                )
                if motion_tracker is not None:
                    if P_raw is not None:
                        motion_tracker.seed(frame, P_raw)
                    else:
                        P_flow = motion_tracker.propagate(frame)
                        if P_flow is not None:
                            player_boxes = [p["bbox"] for p in players]
                            if (check_projection_sanity(P_flow, player_boxes)
                                    and check_line_alignment(P_flow, frame, min_score=0.12)):
                                P_raw = P_flow
                            else:
                                motion_tracker.reset()
                if P_raw is not None:
                    P_raw = refine_projection_to_lines(P_raw, frame)
                P_smooth = smoother.update(P_raw)
                source = "pipeline" if P_smooth is not None else "none"
            homog_source_counts[source] = homog_source_counts.get(source, 0) + 1

        # ── Speed / distance (updated every frame, pre-roll included) ─
        if P_smooth is not None:
            player_pitch_pts = {}
            for p in players:
                tid = p["track_id"]
                if tid < 0:
                    continue
                xy = project_foot(p["bbox"], P_smooth)
                if xy is not None:
                    display_tid = resolve_tid(tid, render_sec)
                    info = PLAYER_NAMES.get(display_tid, (None, 2, None))
                    name, team_id, _ = info
                    # Distance keys on the player NAME so it accumulates
                    # across track-id changes (re-entries, mid-clip remaps).
                    # Unmapped tracks get a per-track key so they still count.
                    canonical = name if name else f"track_{display_tid}"
                    player_stats.update(tid, xy, canonical_id=canonical)
                    player_pitch_pts[display_tid] = (xy, team_id)

        # ── Ball history + velocity extrapolation ────────────────────
        # YOLO loses the ball during dribbles / occlusions. We keep the
        # last 2 detected bboxes and step the prediction forward from
        # them when a frame has no detection, so the demo overlay shows
        # a continuous ball instead of flickering on/off.
        if best_ball_bbox is not None:
            _ball_bboxes.append(best_ball_bbox)
            _ball_frames.append(abs_frame)
            pitch_xy = (project_foot(best_ball_bbox, P_smooth)
                        if P_smooth is not None else None)
            _ball_pitch.append(pitch_xy)
            if len(_ball_bboxes) > 2:
                _ball_bboxes.pop(0)
                _ball_frames.pop(0)
                _ball_pitch.pop(0)

        # Decide what to draw / project this frame, in order of trust:
        #   1. Manually annotated ball position (linearly interpolated). If
        #      the user labelled bracketing frames, this wins — it's the
        #      ground truth and is continuous by construction.
        #   2. YOLO detection on this frame.
        #   3. Velocity-extrapolated YOLO from the last two detections.
        draw_bbox = None
        ball_pitch_pt = None

        ann_pos = interpolate_ball_annotation(ball_annotations, abs_frame)
        if ann_pos is not None:
            ax_, ay_ = int(ann_pos[0]), int(ann_pos[1])
            half = 8   # synthetic bbox half-width — matches a typical ball
                       # footprint and keeps the triangle anchor right
            draw_bbox = [
                int(np.clip(ax_ - half, 0, fw - 1)),
                int(np.clip(ay_ - half, 0, fh_v - 1)),
                int(np.clip(ax_ + half, 0, fw - 1)),
                int(np.clip(ay_ + half, 0, fh_v - 1)),
            ]
            if P_smooth is not None:
                ball_pitch_pt = project_foot(draw_bbox, P_smooth)
        elif best_ball_bbox is not None:
            draw_bbox = best_ball_bbox
            ball_pitch_pt = _ball_pitch[-1] if _ball_pitch else None
        elif (_ball_frames
                and (abs_frame - _ball_frames[-1]) <= BALL_MAX_GAP_FRAMES):
            gap = abs_frame - _ball_frames[-1]
            last = _ball_bboxes[-1]
            if len(_ball_bboxes) == 2:
                dt = _ball_frames[-1] - _ball_frames[-2]
                if dt > 0:
                    # Linear damping: full velocity at gap=1, zero at the
                    # max gap. Prevents a stale prediction from rocketing
                    # off-screen if the ball was last seen mid-pass.
                    damp = 1.0 - gap / (BALL_MAX_GAP_FRAMES + 1)
                    vx = (last[0] - _ball_bboxes[-2][0]) / dt * damp
                    vy = (last[1] - _ball_bboxes[-2][1]) / dt * damp
                    dx = int(vx * gap)
                    dy = int(vy * gap)
                    draw_bbox = [
                        int(np.clip(last[0] + dx, 0, fw - 1)),
                        int(np.clip(last[1] + dy, 0, fh_v - 1)),
                        int(np.clip(last[2] + dx, 0, fw - 1)),
                        int(np.clip(last[3] + dy, 0, fh_v - 1)),
                    ]
                    if P_smooth is not None:
                        ball_pitch_pt = project_foot(draw_bbox, P_smooth)
            else:
                # Only one detection on file — hold the same bbox until a
                # new one comes in or the gap times out.
                draw_bbox = last
                ball_pitch_pt = _ball_pitch[-1] if _ball_pitch else None

        if not is_render:
            continue

        # ── Render ────────────────────────────────────────────────────
        out_frame = frame.copy()

        if args.discover:
            for p in players:
                draw_discover_box(out_frame, p["bbox"], p["track_id"])
            if draw_bbox is not None:
                draw_ball_triangle(out_frame, draw_bbox)
        else:
            if P_smooth is not None:
                project_lines(out_frame, P_smooth, color=LINE_COLOR, thickness=LINE_THICKNESS)
            for p in players:
                raw_tid = p["track_id"]
                display_tid = resolve_tid(raw_tid, render_sec)
                name, team_id, jersey_no = PLAYER_NAMES.get(display_tid, (None, 2, None))
                # Speed is per-track (resets on a new BoT-SORT id is OK —
                # there's no continuous motion across the gap). Distance is
                # per-player so it accumulates across re-entries.
                canonical = name if name else f"track_{display_tid}"
                draw_player_annotation(
                    out_frame, p["bbox"], name, team_id,
                    player_stats.speed(raw_tid),
                    player_stats.dist(canonical),
                    display_tid, jersey_no,
                )
            if draw_bbox is not None:
                draw_ball_triangle(out_frame, draw_bbox)
            if P_smooth is not None and player_pitch_pts:
                viewport = compute_viewport_polygon(P_smooth, fw, fh_v)
                draw_minimap(out_frame, player_pitch_pts, ball_pitch_pt,
                             viewport_pts=viewport)

        draw_timestamp(out_frame, abs_frame, fps)
        writer.write(out_frame)

    cap.release()
    writer.release()
    if homog_source_counts:
        total = sum(homog_source_counts.values())
        print("\nHomography source breakdown:")
        for k, v in sorted(homog_source_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:14s}: {v:4d} ({100*v/max(total,1):5.1f}%)")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
