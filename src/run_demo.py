"""
Demo video for FSCG presentation.

Step 1 — find track IDs in the clip:
    python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16 --discover

Step 2 — edit PLAYER_NAMES below, then render final demo:
    python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16
"""
import argparse
import sys
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
}

# When two players' paths cross, BoT-SORT can swap their IDs.
# Add an entry for each swap: (seconds_into_clip, {old_id: new_id, new_id: old_id})
# Remaps are applied in threshold order, cumulatively — so chained swaps work.
# e.g. at second 8.5, tracks 253 and 89 swapped:
#   TRACK_REMAPS = [(8.5, {253: 89, 89: 253})]
TRACK_REMAPS: list[tuple[float, dict[int, int]]] = [
]


def resolve_tid(tid: int, render_sec: float) -> int:
    """Translate a raw tracker ID to its corrected ID at the given clip time."""
    for threshold, remap in sorted(TRACK_REMAPS, key=lambda x: x[0]):
        if render_sec >= threshold:
            tid = remap.get(tid, tid)
    return tid


PREROLL_SEC = 15            # seconds before start_ts: stabilises tracking + seeds PnLCalib
MINIMAP_W = 260             # minimap width in pixels
MINIMAP_H = int(MINIMAP_W * 68 / 105)
SPEED_EMA = 0.25            # lower = smoother speed readings
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 2


# ── Speed / distance tracker ──────────────────────────────────────────────────

class PlayerStats:
    # Hard clips. 40 km/h is the realistic sprint ceiling (Bolt is ~44).
    # Above TELEPORT_M @ 25fps (~270 km/h) is always a bad projection
    # or an ID swap — snap position, don't accumulate distance or bump speed.
    # NOISE_FLOOR_M absorbs sub-pixel homography jitter so a standing player
    # doesn't rack up distance from small oscillations (observed: ~1.5m/s drift).
    MAX_SPEED_KMH = 40.0
    TELEPORT_M = 3.0
    NOISE_FLOOR_M = 0.12  # anything below this is jitter, not movement

    def __init__(self, fps: float):
        self.fps = fps
        self._pos: dict[int, np.ndarray] = {}
        self._speed: dict[int, float] = {}
        self._dist: dict[int, float] = {}

    def update(self, track_id: int, pitch_xy: np.ndarray) -> None:
        if track_id in self._pos:
            d = float(np.linalg.norm(pitch_xy - self._pos[track_id]))
            if d > self.TELEPORT_M:
                # Bad homography or ID swap.
                self._speed[track_id] = (1 - SPEED_EMA) * self._speed.get(track_id, 0.0)
                self._pos[track_id] = pitch_xy
                return
            if d < self.NOISE_FLOOR_M:
                # Stationary within jitter — decay speed, don't add distance, don't move pos
                # (so small oscillations don't bias the position reference).
                self._speed[track_id] = (1 - SPEED_EMA) * self._speed.get(track_id, 0.0)
                return
            speed_kmh = min(d * self.fps * 3.6, self.MAX_SPEED_KMH)
            prev = self._speed.get(track_id, speed_kmh)
            self._speed[track_id] = SPEED_EMA * speed_kmh + (1 - SPEED_EMA) * prev
            self._dist[track_id] = self._dist.get(track_id, 0.0) + d
        self._pos[track_id] = pitch_xy

    def drop(self, track_id: int) -> None:
        """Forget a track's position so the next update starts fresh (no spurious delta)."""
        self._pos.pop(track_id, None)

    def speed(self, track_id: int) -> float:
        return self._speed.get(track_id, 0.0)

    def dist(self, track_id: int) -> float:
        return self._dist.get(track_id, 0.0)


# ── Wide-angle gate ───────────────────────────────────────────────────────────

def is_wide_shot(players: list, frame_h: int,
                 min_players: int = 7, max_tallest_frac: float = 0.22) -> bool:
    """Is this a wide tactical shot (homography meaningful) or a close-up (not)?

    Cheap — uses only the already-detected player boxes, no extra YOLO pass.
    Close-ups have few players and the biggest box fills a big chunk of the frame.
    """
    if len(players) < min_players:
        return False
    tallest = max((p['bbox'][3] - p['bbox'][1]) / frame_h for p in players)
    return tallest <= max_tallest_frac


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


# ── Minimap ───────────────────────────────────────────────────────────────────

def _mm_pt(x: float, y: float, pad: int) -> tuple[int, int]:
    return (int(x / 105 * MINIMAP_W) + pad, int(y / 68 * MINIMAP_H) + pad)


def draw_minimap(
    frame: np.ndarray,
    player_pts: dict[int, tuple[np.ndarray, int]],
    ball_pt: Optional[np.ndarray],
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
    x1 = fw - cw - 14
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


def draw_player_annotation(frame, bbox, name, team_id, speed_kmh, dist_m, track_id, jersey_no=None) -> None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    color = Config.TEAM_COLORS.get(team_id, (180, 180, 180))
    draw_ellipse(frame, bbox, color)

    # Unlabeled players get ellipse only — no badge, no stats
    if not name:
        return

    label = f"{name},{jersey_no}" if jersey_no is not None else name
    _text_badge(frame, label, cx, y1 - 6, color)

    if speed_kmh >= 0.5:
        spd = f"{speed_kmh:.1f} km/h"
        (sw, _), _ = cv2.getTextSize(spd, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, spd, (cx - sw // 2, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    if dist_m >= 1.0:
        ds = f"{dist_m:.0f}m"
        (dw, _), _ = cv2.getTextSize(ds, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(frame, ds, (cx - dw // 2, y2 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)


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
    if manual_seeds:
        print(f"Loaded {len(manual_seeds)} manual seed(s) for {args.match}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    P_smooth = None
    player_pitch_pts: dict[int, tuple[np.ndarray, int]] = {}
    ball_pitch_pt: Optional[np.ndarray] = None

    for frame_i in tqdm(range(total_frames), desc=mode):
        ret, frame = cap.read()
        if not ret:
            break

        is_render = frame_i >= preroll_frames
        abs_frame = clip_start + frame_i

        # ── Detection + tracking ──────────────────────────────────────
        results = yolo.track(
            frame,
            conf=min(Config.PLAYER_CONF_THRESHOLD, Config.BALL_CONF_THRESHOLD),
            tracker=f"{Config.TRACKER_TYPE}.yaml",
            persist=True,
            verbose=False,
        )[0]
        boxes = results.boxes

        players, balls = [], []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i])
            tid = int(boxes.id[i]) if boxes.id is not None else -1
            det = {"bbox": bbox, "conf": conf, "track_id": tid}
            if cls_id == 0 and conf >= Config.PLAYER_CONF_THRESHOLD:
                players.append(det)
            elif cls_id == 1 and conf >= Config.BALL_CONF_THRESHOLD:
                balls.append(det)

        # ── Homography ────────────────────────────────────────────────
        if pnl_models is not None:
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
                        # Reject drifted flow projections before they reach
                        # the smoother — same gate as run_pnlcalib_video.main().
                        player_boxes = [p["bbox"] for p in players]
                        if (check_projection_sanity(P_flow, player_boxes)
                                and check_line_alignment(P_flow, frame, min_score=0.12)):
                            P_raw = P_flow
                        else:
                            motion_tracker.reset()
            if P_raw is not None:
                P_raw = refine_projection_to_lines(P_raw, frame)
            P_smooth = smoother.update(P_raw)

        # ── Speed / distance (updated every frame, pre-roll included) ─
        if P_smooth is not None:
            player_pitch_pts = {}
            for p in players:
                tid = p["track_id"]
                if tid < 0:
                    continue
                xy = project_foot(p["bbox"], P_smooth)
                if xy is not None:
                    team_id = PLAYER_NAMES.get(tid, (None, 2, None))[1]
                    player_stats.update(tid, xy)
                    player_pitch_pts[tid] = (xy, team_id)

            ball_pitch_pt = None
            if balls:
                ball_pitch_pt = project_foot(balls[0]["bbox"], P_smooth)

        if not is_render:
            continue

        # ── Render ────────────────────────────────────────────────────
        out_frame = frame.copy()

        if args.discover:
            for p in players:
                draw_discover_box(out_frame, p["bbox"], p["track_id"])
            for b in balls:
                draw_ball_triangle(out_frame, b["bbox"])
        else:
            if P_smooth is not None:
                project_lines(out_frame, P_smooth, color=LINE_COLOR, thickness=LINE_THICKNESS)
            render_sec = (frame_i - preroll_frames) / fps
            for p in players:
                tid = resolve_tid(p["track_id"], render_sec)
                name, team_id, jersey_no = PLAYER_NAMES.get(tid, (None, 2, None))
                draw_player_annotation(
                    out_frame, p["bbox"], name, team_id,
                    player_stats.speed(tid), player_stats.dist(tid), tid, jersey_no,
                )
            for b in balls:
                draw_ball_triangle(out_frame, b["bbox"])
            if P_smooth is not None and player_pitch_pts:
                draw_minimap(out_frame, player_pitch_pts, ball_pitch_pt)

        draw_timestamp(out_frame, abs_frame, fps)
        writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
