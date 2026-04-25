"""
Run PnLCalib on a match clip and output video with projected pitch lines.

Includes:
  - Non-gameplay frame filtering: skip close-ups/replays via scoreboard check
  - Player sanity check: reject projections where most players land off-pitch
  - Line alignment check: verify projected lines match visible white pixels
  - Temporal consistency: reject outlier projections that deviate from stable baseline
  - Temporal smoothing: EMA on projection matrix to eliminate jitter

Usage:
    python -m src.run_pnlcalib_video --match dec-mla --offset_min 10 --duration_sec 60
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Resolve project root (src/ is one level down)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PNLCALIB_DIR = PROJECT_ROOT / "models" / "pnlcalib"

# Add PnLCalib to path for model/utils imports
sys.path.insert(0, str(PNLCALIB_DIR))

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

# Add project root to path for src imports
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import Config
from src.camera_motion import CameraMotionTracker
from src.manual_calibration import load_match_seeds

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# ── Pitch geometry for line projection ───────────────────────────────

LINES_COORDS = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]],
    [[0., 37.66, 0.], [0., 37.66, -2.44]],
    [[0., 30.34, 0.], [0., 30.34, -2.44]],
    [[105., 37.66, -2.44], [105., 30.34, -2.44]],
    [[105., 30.34, 0.], [105., 30.34, -2.44]],
    [[105., 37.66, 0.], [105., 37.66, -2.44]],
    [[52.5, 0., 0.], [52.5, 68, 0.]],
    [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]],
    [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]],
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]],
]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params["x_focal_length"]
    y_focal_length = cam_params["y_focal_length"]
    principal_point = np.array(cam_params["principal_point"])
    position_meters = np.array(cam_params["position_meters"])
    rotation = np.array(cam_params["rotation_matrix"])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([
        [x_focal_length, 0, principal_point[0]],
        [0, y_focal_length, principal_point[1]],
        [0, 0, 1],
    ])
    P = Q @ (rotation @ It)
    return P


def _project_point(P, world_pt):
    """Project a 3D world point to image coordinates. Returns (x, y) or None if behind camera."""
    p = P @ np.array([world_pt[0] - 52.5, world_pt[1] - 34, world_pt[2], 1])
    if p[2] <= 0:
        return None
    return p[0] / p[2], p[1] / p[2]


def _clamp_int(v, lo=-100000, hi=100000):
    """Clamp to safe int range for OpenCV drawing."""
    return int(max(lo, min(hi, round(v))))


def _draw_sampled_line(frame, P, world_a, world_b, n_samples, color, thickness):
    """
    Project a world-space line by sampling intermediate points.
    Handles lines whose endpoints are far off-screen (e.g., touchlines)
    by drawing only the segments that pass through/near the frame.
    """
    h, w = frame.shape[:2]
    limit = max(w, h) * 4  # points beyond this are off-screen noise

    pts = []
    for t in np.linspace(0, 1, n_samples):
        wx = world_a[0] + t * (world_b[0] - world_a[0])
        wy = world_a[1] + t * (world_b[1] - world_a[1])
        wz = world_a[2] + t * (world_b[2] - world_a[2])
        ip = _project_point(P, (wx, wy, wz))
        if ip is None or abs(ip[0]) > limit or abs(ip[1]) > limit:
            # Behind camera or way off-screen — flush current segment
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)
            pts = []
            continue
        pts.append([_clamp_int(ip[0]), _clamp_int(ip[1])])
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)


def project_lines(frame, P, color=(0, 0, 255), thickness=2):
    """Project pitch lines onto a frame using the 3x4 projection matrix."""
    for line in LINES_COORDS:
        length = np.sqrt(sum((a - b) ** 2 for a, b in zip(line[0], line[1])))
        n_samples = max(2, int(length / 5))  # ~1 sample per 5 meters
        _draw_sampled_line(frame, P, line[0], line[1], n_samples, color, thickness)

    # Circles and arcs — sample points along arc, draw as polyline
    h, w = frame.shape[:2]
    limit = max(w, h) * 4
    r = 9.15
    for base_world, ang_range in [
        ((11.0, 34.0, 0.),  (37, 143, 50)),      # left penalty arc
        ((94.0, 34.0, 0.),  (217, 323, 200)),     # right penalty arc
        ((52.5, 34.0, 0.),  (0, 360, 500)),       # center circle
    ]:
        pts = []
        for ang in np.linspace(*ang_range):
            ang_rad = np.deg2rad(ang)
            world = (
                base_world[0] + r * np.sin(ang_rad),
                base_world[1] + r * np.cos(ang_rad),
                0.0,
            )
            ip = _project_point(P, world)
            if ip is None or abs(ip[0]) > limit or abs(ip[1]) > limit:
                if len(pts) > 1:
                    cv2.polylines(frame, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)
                pts = []
                continue
            pts.append([_clamp_int(ip[0]), _clamp_int(ip[1])])
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)

    return frame


# ── Non-gameplay frame detection ────────────────────────────────────

def is_gameplay_frame(frame, scoreboard_roi, edge_threshold=0.08):
    """
    Quick check: is the scoreboard visible? If not, this is likely a
    replay, close-up, or transition frame where homography is meaningless.

    Uses the same edge density approach as broadcast.py's check_scoreboard.
    Gameplay frames have the scoreboard overlay which creates consistent
    edges in the ROI. Returns False for replays/close-ups/graphics.
    """
    x1, y1, x2, y2 = scoreboard_roi
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.mean(edges > 0))
    return density >= edge_threshold


def green_ratio(frame):
    """Fraction of the frame that looks like pitch grass.

    Wide tactical shots typically sit 40–70% green. Medium tight shots
    (one PA + a couple of players) 25–40%. Close-ups, dugout shots,
    replays, and crowd shots are <15%. This is the cleanest signal for
    "is the camera zoomed out enough for homography to make sense."
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 40)
    )
    return float(np.mean(mask))


def yolo_shot_features(frame, yolo):
    """Detect players and summarize shot framing.

    Returns ``(n_players, tallest_frac)`` where:
      - ``n_players`` is the count of YOLO person detections
      - ``tallest_frac`` is the height of the biggest player box as a
        fraction of frame height.

    Wide tactical shots: n_players ≈ 10–22, tallest ≈ 0.04–0.10.
    Close-ups / side-camera portraits: n_players ≤ 4, tallest ≥ 0.25.
    """
    h = frame.shape[0]
    n = 0
    tallest = 0.0
    for r in yolo(frame, verbose=False):
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            n += 1
            y1, y2 = float(box.xyxy[0][1]), float(box.xyxy[0][3])
            tallest = max(tallest, (y2 - y1) / h)
    return n, tallest


def is_wide_angle_frame(frame, yolo, scoreboard_roi,
                        scoreboard_threshold=0.08,
                        min_green=0.30,
                        min_players=7,
                        max_tallest_frac=0.22):
    """
    Combined wide-angle gameplay detector.

    Requires:
      - scoreboard overlay present (gameplay, not a replay),
      - enough green pitch in view (not a dugout/crowd shot),
      - many small players detected (not a close-up of one or two).

    The player-count + max-height check is what catches side-camera
    close-ups with a green-kitted goalkeeper or a single player walking
    where the grass background would otherwise fool the green-ratio test.
    """
    if not is_gameplay_frame(frame, scoreboard_roi, scoreboard_threshold):
        return False
    if green_ratio(frame) < min_green:
        return False
    n, tallest = yolo_shot_features(frame, yolo)
    return n >= min_players and tallest <= max_tallest_frac


# ── Sanity checks ───────────────────────────────────────────────────

def check_projection_sanity(P, player_boxes, min_players=5, min_on_pitch_ratio=0.5, margin=5.0):
    """
    Reject projections where most detected players don't land on the pitch.
    Projects foot position (bottom-center of bbox) to pitch coords.
    Returns True if projection looks plausible.
    """
    if len(player_boxes) < min_players:
        return False

    H_img_to_pitch = np.linalg.inv(P[:, [0, 1, 3]])
    on_pitch = 0
    for x1, y1, x2, y2 in player_boxes:
        foot_x = (x1 + x2) / 2
        foot_y = y2
        pitch_pt = H_img_to_pitch @ np.array([foot_x, foot_y, 1.0])
        if abs(pitch_pt[2]) < 1e-10:
            continue
        px = pitch_pt[0] / pitch_pt[2] + 52.5
        py = pitch_pt[1] / pitch_pt[2] + 34.0
        if -margin <= px <= PITCH_LENGTH + margin and -margin <= py <= PITCH_WIDTH + margin:
            on_pitch += 1

    ratio = on_pitch / len(player_boxes)
    return ratio >= min_on_pitch_ratio


def _pitch_masks(frame):
    """Build dilated white-line and green-pitch masks from a BGR frame.

    Returns (white_mask, green_near_mask), both uint8 in {0,1}, shape (H, W).

    ``white_mask``: bright, low-saturation pixels (paint + scoreboard
    graphics + stand paint). Dilated generously so nearby hits count.

    ``green_near_mask``: pitch grass dilated so a pixel adjacent to
    green (including lines painted on grass) counts as on-pitch.
    Running tracks, stands, and buildings are not green, so they
    won't satisfy this mask — which is what lets us reject projections
    that put the touchline on a running track or in the crowd.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 140)).astype(np.uint8)
    white = cv2.dilate(white, np.ones((11, 11), np.uint8), iterations=2)

    # Wide green band — handles dark/dry/wet/yellowed grass. S must be
    # moderate so we don't grab gray tarmac or dusty running track.
    green = (
        (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 40)
    ).astype(np.uint8)
    # Close small gaps (players, shadows) then dilate so pixels adjacent
    # to the pitch (lines painted on grass) count as on-pitch. Dilated
    # generously — we only need to distinguish "near pitch" from "in the
    # stands / on the running track".
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    green_near = cv2.dilate(green, np.ones((11, 11), np.uint8), iterations=3)

    return white, green_near


def check_line_alignment(P, frame, min_score=0.12, min_per_line_on_pitch=0.55,
                         min_on_pitch_overall=0.45):
    """
    Verify that projected pitch lines lie on painted lines that sit on
    or next to actual pitch grass.

    Gates:
      1. **Per-line green-adjacency (strict)**: each touchline / halfway
         line / PA edge that is mostly inside the frame must have at
         least ``min_per_line_on_pitch`` of its sampled points on or
         next to green pitch. This is the check that rejects
         "touchline-drawn-through-the-bleachers" and
         "touchline-drawn-on-the-running-track" — failure modes where
         the overall projection looks plausible on average but one
         specific line is clearly off the pitch.

      2. **Overall green-adjacency**: ``min_on_pitch_overall`` of all
         samples on green. Redundant with (1) but guards against cases
         where every individual line is marginally off.

      3. **Paint alignment**: ``min_score`` of samples must land on
         bright (white) pixels. Ensures the projection actually follows
         painted lines rather than random empty grass.
    """
    h, w = frame.shape[:2]
    white_mask, green_near = _pitch_masks(frame)

    # Touchlines + halfway line + PA edges — the lines most likely to
    # end up visibly off-pitch when PnLCalib mis-scales the camera.
    key_lines = [
        [[0., 68., 0.], [105., 68., 0.]],      # near touchline
        [[0., 0., 0.], [105., 0., 0.]],         # far touchline
        [[52.5, 0., 0.], [52.5, 68, 0.]],       # halfway line
        [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], # left PA side
        [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], # right PA side
        [[0., 54.16, 0.], [16.5, 54.16, 0.]],   # left PA top
        [[88.5, 54.16, 0.], [105., 54.16, 0.]], # right PA top
    ]

    total_hits = 0
    total_on_pitch = 0
    total_samples = 0
    bad_lines = 0  # lines that land almost entirely off-grass
    lines_checked = 0

    for line in key_lines:
        length = np.sqrt(sum((a - b) ** 2 for a, b in zip(line[0], line[1])))
        n = max(3, int(length / 8))

        line_in = 0
        line_on_pitch = 0
        line_on_white = 0
        for t in np.linspace(0, 1, n):
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            ip = _project_point(P, (wx, wy, 0.0))
            if ip is None:
                continue
            ix, iy = int(round(ip[0])), int(round(ip[1]))
            if 0 <= ix < w and 0 <= iy < h:
                line_in += 1
                if green_near[iy, ix]:
                    line_on_pitch += 1
                    if white_mask[iy, ix]:
                        line_on_white += 1

        # Only lines with a reasonable number of samples actually inside
        # the frame are informative. Skip lines that are mostly off-screen
        # (e.g. far touchline on a tight shot of one PA).
        if line_in >= 3:
            lines_checked += 1
            if (line_on_pitch / line_in) < min_per_line_on_pitch:
                bad_lines += 1

        total_samples += line_in
        total_on_pitch += line_on_pitch
        total_hits += line_on_white

    if total_samples < 5 or lines_checked < 2:
        return False
    if bad_lines >= 1:
        # Any single line living off the pitch is usually a giveaway —
        # halfway line stretching into the crowd, touchline on the track.
        return False
    if (total_on_pitch / total_samples) < min_on_pitch_overall:
        return False
    return (total_hits / total_samples) >= min_score


# ── Line-snap refinement ────────────────────────────────────────────

# Planar (z=0) lines used as refinement anchors. Subset of LINES_COORDS —
# skips the goal-post verticals (z!=0) since the refined H is planar.
_REFINE_LINES = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],      # left PA top
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],    # left PA side
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],      # left PA bottom
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],    # right PA top
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],    # right PA side
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],    # right PA bottom
    [[52.5, 0., 0.], [52.5, 68., 0.]],         # halfway line
    [[0., 68., 0.], [105., 68., 0.]],          # near touchline
    [[0., 0., 0.], [105., 0., 0.]],            # far touchline
    [[0., 0., 0.], [0., 68., 0.]],             # left goal line
    [[105., 0., 0.], [105., 68., 0.]],         # right goal line
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],       # left 6yd top
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],      # left 6yd side
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],       # left 6yd bottom
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],    # right 6yd top
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],    # right 6yd side
    [[99.5, 24.84, 0.], [105., 24.84, 0.]],    # right 6yd bottom
]


REFINE_STATS = {"attempts": 0, "applied": 0, "rejected_worse": 0, "too_few": 0}


def _line_alignment_score(P, frame):
    """Fraction of sampled pitch-line points that land on painted-on-pitch pixels.

    Same sampling scheme as check_line_alignment's third gate, but returns the
    ratio rather than a pass/fail so we can compare two projections.
    """
    h, w = frame.shape[:2]
    white_mask, green_near = _pitch_masks(frame)
    total_samples = 0
    total_hits = 0
    for line in _REFINE_LINES:
        world_len = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(line[0], line[1]))))
        n = max(3, int(world_len / 2.0))
        for t in np.linspace(0, 1, n):
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            ip = _project_point(P, (wx, wy, 0.0))
            if ip is None:
                continue
            ix, iy = int(round(ip[0])), int(round(ip[1]))
            if 0 <= ix < w and 0 <= iy < h:
                total_samples += 1
                if white_mask[iy, ix] and green_near[iy, ix]:
                    total_hits += 1
    if total_samples < 10:
        return 0.0
    return total_hits / total_samples


def refine_projection_to_lines(P, frame, search_radius=18, min_inliers=25,
                                ransac_thresh_px=3.0):
    """Snap a projection's pitch-lines to painted white pixels in the frame.

    For each planar pitch line, sample points in world space, project into the
    image, and search perpendicular to the line for the nearest bright pixel
    that also sits on (or next to) green pitch. Collected correspondences are
    fed to cv2.findHomography + RANSAC to re-estimate the image -> pitch
    plane homography, which is packed back into a 3x4 P (z-column zero).

    Returns the refined 3x4 P, or the input P unchanged when:
      - too few correspondences are found (drift is too large or frame has
        few painted lines visible), or
      - the refined projection shifts a pitch-center reference point by more
        than ``max_shift_px`` (guards against RANSAC picking up a plausible
        but wrong transform from stadium paint).

    Requires the input P to already be close enough that projected lines
    land within ``search_radius`` px of the painted ones — i.e. it corrects
    residual drift, it can't rescue a grossly wrong projection.
    """
    REFINE_STATS["attempts"] += 1
    h, w = frame.shape[:2]
    white_mask, green_near = _pitch_masks(frame)

    world_xy = []
    img_xy = []

    for line in _REFINE_LINES:
        # Project endpoints to get line direction in image space.
        ip_a = _project_point(P, line[0])
        ip_b = _project_point(P, line[1])
        if ip_a is None or ip_b is None:
            continue
        dx, dy = ip_b[0] - ip_a[0], ip_b[1] - ip_a[1]
        seg_len = float(np.hypot(dx, dy))
        if seg_len < 12:
            continue
        # Unit perpendicular to the projected line.
        px_u, py_u = -dy / seg_len, dx / seg_len

        world_len = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(line[0], line[1]))))
        n_samples = max(3, int(world_len / 1.5))  # ~1 sample per 1.5 m

        for t in np.linspace(0, 1, n_samples):
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            ip = _project_point(P, (wx, wy, 0.0))
            if ip is None:
                continue
            ix, iy = ip[0], ip[1]
            # Skip samples far outside the visible frame.
            if not (-20 <= ix < w + 20 and -20 <= iy < h + 20):
                continue

            # Walk out from the projected point along the perpendicular,
            # stop at the first painted-on-pitch pixel we hit.
            best = None
            for d in range(0, search_radius + 1):
                for sign in (1, -1) if d > 0 else (1,):
                    sx = int(round(ix + sign * d * px_u))
                    sy = int(round(iy + sign * d * py_u))
                    if 0 <= sx < w and 0 <= sy < h:
                        if white_mask[sy, sx] and green_near[sy, sx]:
                            best = (sx, sy)
                            break
                if best is not None:
                    break
            if best is not None:
                world_xy.append((wx - 52.5, wy - 34.0))  # origin-centered
                img_xy.append(best)

    if len(world_xy) < min_inliers:
        REFINE_STATS["too_few"] += 1
        return P

    img_arr = np.asarray(img_xy, dtype=np.float32)
    world_arr = np.asarray(world_xy, dtype=np.float32)
    H_img_to_pitch, mask = cv2.findHomography(
        img_arr, world_arr, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_px,
    )
    if H_img_to_pitch is None or mask is None or int(mask.sum()) < min_inliers:
        REFINE_STATS["too_few"] += 1
        return P

    from src.camera_motion import projection_from_plane_homography
    P_refined = projection_from_plane_homography(H_img_to_pitch.astype(np.float64))

    # Only accept the refinement if projected pitch lines now lie on paint
    # at least as well as they did before. Catches the RANSAC-found-a-
    # self-consistent-but-wrong-fit case (snapping halfway-line samples onto
    # the far-touchline paint, etc.) without blocking large, correct pan
    # corrections.
    score_in = _line_alignment_score(P, frame)
    score_out = _line_alignment_score(P_refined, frame)
    if score_out + 0.02 < score_in:
        REFINE_STATS["rejected_worse"] += 1
        return P

    REFINE_STATS["applied"] += 1
    return P_refined


# ── Temporal smoothing with outlier rejection ───────────────────────

class ProjectionSmoother:
    """
    Exponential moving average on the 3x4 projection matrix with
    outlier rejection.

    Three modes:
    - Small change (normal camera movement): EMA blend for smooth transitions
    - Medium change: only accept if confirmed by multiple consecutive frames
    - Large change (camera cut): reset immediately

    This prevents single-frame false positives from corrupting the
    stable projection (the exact problem shown in dec-mla screenshots).
    """
    def __init__(self, alpha=0.3, max_stale_frames=75,
                 reset_threshold=0.5, outlier_threshold=0.2,
                 confirm_frames=3):
        self.alpha = alpha
        self.max_stale = max_stale_frames
        self.reset_threshold = reset_threshold
        self.outlier_threshold = outlier_threshold  # diff above this needs confirmation
        self.confirm_needed = confirm_frames         # how many consecutive deviations before accepting
        self.smoothed_P = None
        self.stale_count = 0
        self.resets = 0
        self.outlier_rejected = 0
        # Track consecutive deviating frames for confirmation
        self._pending_P = None
        self._pending_count = 0

    def _norm_diff(self, P_new):
        """Normalized difference between P_new and current smoothed P."""
        P_norm = P_new / np.linalg.norm(P_new)
        S_norm = self.smoothed_P / np.linalg.norm(self.smoothed_P)
        if np.sum(P_norm * S_norm) < 0:
            P_norm = -P_norm
        return np.linalg.norm(P_norm - S_norm), P_norm

    def update(self, P_new):
        """
        Feed a new projection matrix. Returns the smoothed version,
        or None if we have no usable projection.
        """
        if P_new is None:
            self._pending_P = None
            self._pending_count = 0
            if self.smoothed_P is not None and self.stale_count < self.max_stale:
                self.stale_count += 1
                return self.smoothed_P.copy()
            return None

        if self.smoothed_P is None:
            self.smoothed_P = P_new.copy()
            self.stale_count = 0
            return self.smoothed_P.copy()

        # Handle sign ambiguity
        if np.sum(P_new / np.linalg.norm(P_new) * self.smoothed_P / np.linalg.norm(self.smoothed_P)) < 0:
            P_new = -P_new

        diff, _ = self._norm_diff(P_new)

        if diff > self.reset_threshold:
            # Very large change — likely a camera cut.
            # But require confirmation to avoid single-frame false resets.
            if self._pending_P is not None:
                pending_diff = np.linalg.norm(
                    P_new / np.linalg.norm(P_new) - self._pending_P / np.linalg.norm(self._pending_P)
                )
                if pending_diff < self.outlier_threshold:
                    # New frame agrees with the pending direction
                    self._pending_count += 1
                    self._pending_P = P_new.copy()
                    if self._pending_count >= self.confirm_needed:
                        # Confirmed: this is a real change (camera cut/pan)
                        self.smoothed_P = P_new.copy()
                        self.resets += 1
                        self._pending_P = None
                        self._pending_count = 0
                        self.stale_count = 0
                        return self.smoothed_P.copy()
                else:
                    # Doesn't agree with pending — start new pending
                    self._pending_P = P_new.copy()
                    self._pending_count = 1
            else:
                self._pending_P = P_new.copy()
                self._pending_count = 1

            # While awaiting confirmation, reuse last good projection
            self.outlier_rejected += 1
            self.stale_count += 1
            if self.stale_count < self.max_stale:
                return self.smoothed_P.copy()
            return None

        elif diff > self.outlier_threshold:
            # Medium change — suspicious. Could be a momentary glitch
            # or the start of a real camera movement.
            if self._pending_P is not None:
                pending_diff = np.linalg.norm(
                    P_new / np.linalg.norm(P_new) - self._pending_P / np.linalg.norm(self._pending_P)
                )
                if pending_diff < self.outlier_threshold:
                    self._pending_count += 1
                    self._pending_P = P_new.copy()
                    if self._pending_count >= self.confirm_needed:
                        # Confirmed medium shift — blend toward it
                        self.smoothed_P = self.alpha * P_new + (1 - self.alpha) * self.smoothed_P
                        self._pending_P = None
                        self._pending_count = 0
                        self.stale_count = 0
                        return self.smoothed_P.copy()
                else:
                    self._pending_P = P_new.copy()
                    self._pending_count = 1
            else:
                self._pending_P = P_new.copy()
                self._pending_count = 1

            # Reject this frame, reuse stable projection
            self.outlier_rejected += 1
            self.stale_count += 1
            if self.stale_count < self.max_stale:
                return self.smoothed_P.copy()
            return None

        else:
            # Small change — normal camera movement. Blend smoothly.
            self._pending_P = None
            self._pending_count = 0
            self.smoothed_P = self.alpha * P_new + (1 - self.alpha) * self.smoothed_P
            self.stale_count = 0
            return self.smoothed_P.copy()


# ── PnLCalib inference ───────────────────────────────────────────────

def run_inference(cam, frame_bgr, model_kp, model_line, device, transform_resize,
                  kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=True):
    """Run PnLCalib inference on a single frame, return camera params or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    frame_t = tvf.to_tensor(frame_pil).float().unsqueeze(0)
    if frame_t.size()[-1] != 960:
        frame_t = transform_resize(frame_t)
    frame_t = frame_t.to(device)
    _, _, h_t, w_t = frame_t.size()

    with torch.no_grad():
        heatmaps = model_kp(frame_t)
        heatmaps_l = model_line(frame_t)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w_t, h=h_t, normalize=True)

    cam.update(kp_dict, lines_dict)
    try:
        final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)
    except cv2.error:
        # PnLCalib internally calls cv2.calibrateCamera which occasionally
        # asserts on degenerate keypoint configurations. Treat as inference
        # failure rather than crashing the whole clip.
        return None

    return final_params_dict


# ── Reusable helpers for library use (notebook, cross-match inspection) ─

def load_models(device: str = "cuda:0") -> dict:
    """Load PnLCalib keypoint + line models + YOLO + resize transform.

    Returns a dict suitable for passing to ``predict_one_frame`` /
    ``inspect_match``. Loading takes ~3s and ~800MB VRAM, so cache this
    across loops (do it once per notebook session).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(str(PNLCALIB_DIR / "config" / "hrnetv2_w48.yaml"), "r"))
    cfg_l = yaml.safe_load(open(str(PNLCALIB_DIR / "config" / "hrnetv2_w48_l.yaml"), "r"))

    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(torch.load(str(PNLCALIB_DIR / "weights" / "SV_kp"),
                                        map_location=device))
    model_kp.to(device).eval()

    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(torch.load(str(PNLCALIB_DIR / "weights" / "SV_lines"),
                                          map_location=device))
    model_line.to(device).eval()

    yolo = YOLO(str(Config.resolve_yolo_model()))

    return {
        "device": device,
        "model_kp": model_kp,
        "model_line": model_line,
        "yolo": yolo,
        "transform_resize": T.Resize((540, 960)),
    }


def predict_one_frame(
    frame: np.ndarray,
    models: dict,
    frame_w: int,
    frame_h: int,
    manual_seeds: Optional[dict] = None,
    abs_frame: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Run PnLCalib + sanity gates + manual-seed fallback on a single frame.

    No optical flow, no smoothing — each call is independent. Returns
    ``(P_3x4_or_None, status)`` where ``status`` is one of:

      - ``pnlcalib``      - PnLCalib succeeded + passed both gates
      - ``manual_seed``   - PnLCalib failed but a manual seed passed gates
      - ``sanity_fail``   - PnLCalib output rejected (players off pitch)
      - ``line_fail``     - PnLCalib output rejected (lines didn't hug paint)
      - ``inference_fail``- PnLCalib returned nothing
    """
    device = models["device"]

    player_boxes = []
    for r in models["yolo"](frame, verbose=False):
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                player_boxes.append((float(x1), float(y1), float(x2), float(y2)))

    # Try default thresholds first; if PnLCalib returns nothing or its
    # output fails the gates, retry once with looser keypoint/line
    # thresholds. This recovers marginal wide-angle frames where paint
    # contrast is low (overcast pitches, pale lines) without lowering
    # the overall confidence bar.
    status = "inference_fail"
    best_P = None
    for kp_thr, line_thr in [(0.3434, 0.7867), (0.20, 0.50)]:
        cam = FramebyFrameCalib(iwidth=frame_w, iheight=frame_h, denormalize=True)
        params = run_inference(
            cam, frame, models["model_kp"], models["model_line"],
            device, models["transform_resize"],
            kp_threshold=kp_thr, line_threshold=line_thr,
        )
        if params is None:
            continue
        P = projection_from_cam_params(params)
        if not check_projection_sanity(P, player_boxes):
            status = "sanity_fail"
            continue
        if not check_line_alignment(P, frame, min_score=0.12):
            status = "line_fail"
            continue
        best_P = P
        break

    if best_P is not None:
        return best_P, "pnlcalib"

    # PnLCalib failed — try the nearest manual seed (if any).
    if manual_seeds and abs_frame is not None:
        nearest_f = min(manual_seeds.keys(), key=lambda f: abs(f - abs_frame))
        P_manual = manual_seeds[nearest_f]
        if (check_projection_sanity(P_manual, player_boxes)
                and check_line_alignment(P_manual, frame, min_score=0.12)):
            return P_manual, "manual_seed"

    return None, status


def inspect_match(
    slug: str,
    models: dict,
    n_frames: int = 12,
    offset_min: float = 10.0,
    duration_sec: int = 60,
    out_dir: Optional[Path] = None,
) -> dict:
    """Sample ``n_frames`` uniformly from a clip, predict each, save a grid PNG.

    Fast across-match inspection: no MP4, no flow, no smoothing.
    Result lives at ``out_dir/{slug}_inspect.png`` alongside
    ``{slug}_inspect.stats.json``.

    Returns a dict with the per-frame statuses and the summary counts.
    """
    import matplotlib.pyplot as plt

    out_dir = out_dir or Config.OUTPUT_HOMOGRAPHY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    periods_file = PROJECT_ROOT / "data" / "period_detection_results.json"
    periods = {r["slug"]: r for r in json.loads(periods_file.read_text())
               if r["status"] == "ok"}
    if slug not in periods:
        raise KeyError(f"Match '{slug}' not found in period data")

    fh_start = periods[slug]["first_half_start_frame"]
    fps = periods[slug]["fps"]
    start_frame = fh_start + int(offset_min * 60 * fps)
    total_clip_frames = int(duration_sec * fps)

    video_path = str(Config.MATCH_VIDEOS[slug])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manual_seeds = load_match_seeds(slug)

    # Oversample ~4x candidates uniformly across the clip, then pick the
    # ``n_frames`` most wide-angle gameplay ones. Close-ups, replays, and
    # tight tactical shots are worthless for inspecting homography, so we
    # want the grid to show only frames where a projection is even possible.
    n_candidates = max(n_frames * 4, 40)
    ts = np.linspace(0.02, 0.98, n_candidates)
    offsets = [int(t * total_clip_frames) for t in ts]

    # Pass 1a: cheap pre-filter on every candidate (scoreboard overlay +
    # green ratio). This quickly rules out replays, graphics, dugout
    # shots, crowd shots, so we don't run YOLO on them.
    sx1, sy1, sx2, sy2 = Config.SCOREBOARD_ROI
    candidates = []
    densities = []
    for off in offsets:
        abs_f = start_frame + off
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_f)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = float(np.mean(edges > 0))
        g_ratio = green_ratio(frame)
        candidates.append({
            "abs_frame": abs_f, "frame": frame,
            "scoreboard_density": density,
            "green_ratio": g_ratio,
            "n_players": None,
            "tallest_frac": None,
        })
        densities.append(density)
    cap.release()

    # Calibrate scoreboard threshold (most samples should be gameplay,
    # so 40% of the median is a safe cut-off).
    gameplay_threshold = float(np.median(densities)) * 0.4 if densities else 0.08

    # Pass 1b: YOLO on the pre-passers. Player count + tallest-player
    # height separate wide tactical shots from side-camera close-ups
    # where grass behind a single player would otherwise fool the
    # green-ratio test.
    yolo = models["yolo"]
    for c in candidates:
        if (c["scoreboard_density"] < gameplay_threshold
                or c["green_ratio"] < 0.30):
            continue
        n, tallest = yolo_shot_features(c["frame"], yolo)
        c["n_players"] = n
        c["tallest_frac"] = tallest

    # Wide = scoreboard + green + many small players.
    def _is_wide(c):
        return (
            c["scoreboard_density"] >= gameplay_threshold
            and c["green_ratio"] >= 0.30
            and c["n_players"] is not None
            and c["n_players"] >= 7
            and c["tallest_frac"] is not None
            and c["tallest_frac"] <= 0.22
        )

    wide = [c for c in candidates if _is_wide(c)]

    # Pick ``n_frames`` evenly-spaced wide-angle frames across the clip.
    # Evenly-spaced rather than "greediest green" so the grid still covers
    # the whole time window rather than clustering on one play.
    if len(wide) >= n_frames:
        idx = np.linspace(0, len(wide) - 1, n_frames).round().astype(int)
        wide.sort(key=lambda c: c["abs_frame"])
        chosen = [wide[i] for i in idx]
    else:
        # Not enough wide frames — pad with the greenest non-wide ones so
        # the grid still renders rather than crashing. These will show up
        # as "non_gameplay" in the output.
        chosen = list(wide)
        leftovers = sorted(
            [c for c in candidates if c not in wide],
            key=lambda c: -c["green_ratio"],
        )
        chosen += leftovers[: n_frames - len(wide)]
        chosen.sort(key=lambda c: c["abs_frame"])

    # Pass 2: classify and predict on the chosen frames.
    results = []
    for c in chosen:
        abs_f, frame = c["abs_frame"], c["frame"]
        if not _is_wide(c):
            results.append({"abs_frame": abs_f, "status": "non_gameplay", "frame": frame})
            continue

        P, status = predict_one_frame(frame, models, frame_w, frame_h,
                                      manual_seeds=manual_seeds, abs_frame=abs_f)
        if P is not None:
            frame = project_lines(frame.copy(), P, color=(0, 0, 255), thickness=2)
        results.append({"abs_frame": abs_f, "status": status, "frame": frame})

    # Render the grid.
    cols = 4
    rows = int(np.ceil(n_frames / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), facecolor="white")
    axes = np.atleast_2d(axes).flatten()
    status_color = {
        "pnlcalib":       "green",
        "manual_seed":    "blue",
        "non_gameplay":   "orange",
        "sanity_fail":    "red",
        "line_fail":      "red",
        "inference_fail": "gray",
        "read_fail":      "gray",
    }
    for ax, r in zip(axes, results):
        if "frame" in r:
            ax.imshow(cv2.cvtColor(r["frame"], cv2.COLOR_BGR2RGB))
        ax.set_title(f"f{r['abs_frame']}  {r['status']}",
                     fontsize=9, color=status_color.get(r["status"], "black"))
        ax.axis("off")
    for ax in axes[len(results):]:
        ax.axis("off")

    counts = {k: 0 for k in status_color}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    ok_count         = counts.get("pnlcalib", 0) + counts.get("manual_seed", 0)
    non_gameplay     = counts.get("non_gameplay", 0)
    gameplay_total   = n_frames - non_gameplay
    failed           = gameplay_total - ok_count
    title_color = (
        "green"  if gameplay_total > 0 and ok_count == gameplay_total else
        "orange" if ok_count >= max(gameplay_total, 1) / 2 else
        "red"
    )
    fig.suptitle(
        f"{slug}  |  {ok_count}/{gameplay_total} gameplay OK  "
        f"(pnlcalib={counts.get('pnlcalib', 0)}, manual={counts.get('manual_seed', 0)}, "
        f"failed={failed})  |  {non_gameplay} non-gameplay  |  "
        f"{len(manual_seeds)} seed(s)",
        fontsize=13, fontweight="bold", color=title_color, y=1.00,
    )
    fig.tight_layout()

    png_path = out_dir / f"{slug}_inspect.png"
    fig.savefig(png_path, dpi=90, bbox_inches="tight")
    plt.close(fig)

    stats_path = out_dir / f"{slug}_inspect.stats.json"
    stats_payload = {
        "slug":          slug,
        "n_frames":      n_frames,
        "offset_min":    offset_min,
        "duration_sec":  duration_sec,
        "start_frame":   start_frame,
        "manual_seeds":  len(manual_seeds),
        "status_counts": counts,
        "ok_count":      ok_count,
        "per_frame": [
            {"abs_frame": r["abs_frame"], "status": r["status"]}
            for r in results
        ],
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2))

    return {"png_path": png_path, "stats_path": stats_path, **stats_payload}


def main():
    parser = argparse.ArgumentParser(description="Run PnLCalib on a match clip")
    parser.add_argument("--match", type=str, default="dec-mla", help="Match slug")
    parser.add_argument("--offset_min", type=float, default=10,
                        help="Minutes into the first half to start the clip")
    parser.add_argument("--duration_sec", type=int, default=60, help="Clip duration in seconds")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="EMA smoothing factor (0.1=very smooth, 0.5=responsive)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_flow", action="store_true",
                        help="Disable optical-flow propagation of the last good projection")
    parser.add_argument("--no_manual_seeds", action="store_true",
                        help="Disable manual keypoint seed frames for this match")
    parser.add_argument("--max_flow_streak", type=int, default=60,
                        help="Max consecutive flow-propagated frames before forcing a PnLCalib re-anchor")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load period data to find first-half start ────────────────────
    periods_file = PROJECT_ROOT / "data" / "period_detection_results.json"
    periods = {r["slug"]: r for r in json.loads(periods_file.read_text()) if r["status"] == "ok"}
    if args.match not in periods:
        print(f"Match '{args.match}' not found in period data")
        return
    p = periods[args.match]
    fps = p["fps"]
    fh_start = p["first_half_start_frame"]

    # ── Open video ───────────────────────────────────────────────────
    video_path = str(Config.MATCH_VIDEOS[args.match])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = fh_start + int(args.offset_min * 60 * fps)
    total_frames = int(args.duration_sec * fps)

    print(f"Match: {args.match} | {frame_w}x{frame_h} @ {fps}fps")
    print(f"Clip: frame {start_frame} -> {start_frame + total_frames} ({args.duration_sec}s)")
    print(f"Smoothing alpha: {args.alpha}")

    # ── Calibrate scoreboard threshold on first gameplay frame ───────
    # Sample a few frames to find the edge density of a scoreboard frame
    print("Calibrating scoreboard threshold...")
    scoreboard_roi = Config.SCOREBOARD_ROI
    sample_densities = []
    for offset in range(0, min(250, total_frames), 25):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
        ret, sample_frame = cap.read()
        if ret:
            x1, y1, x2, y2 = scoreboard_roi
            crop = sample_frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            sample_densities.append(float(np.mean(edges > 0)))
    if sample_densities:
        # Use 40% of the median as threshold (scoreboard frames are consistently high)
        median_density = float(np.median(sample_densities))
        scoreboard_threshold = median_density * 0.4
        print(f"  Scoreboard edge density: median={median_density:.4f}, threshold={scoreboard_threshold:.4f}")
    else:
        scoreboard_threshold = 0.08
        print(f"  Using default threshold: {scoreboard_threshold}")

    # ── Load YOLO for player detection (sanity check) ────────────────
    print("Loading YOLO model...")
    yolo = YOLO(str(Config.resolve_yolo_model()))

    # ── Load PnLCalib models ────────────────────────────────────────
    print("Loading PnLCalib models...")
    cfg = yaml.safe_load(open(str(PNLCALIB_DIR / "config" / "hrnetv2_w48.yaml"), "r"))
    cfg_l = yaml.safe_load(open(str(PNLCALIB_DIR / "config" / "hrnetv2_w48_l.yaml"), "r"))

    weights_kp = str(PNLCALIB_DIR / "weights" / "SV_kp")
    weights_line = str(PNLCALIB_DIR / "weights" / "SV_lines")

    loaded_state = torch.load(weights_kp, map_location=device)
    model_kp = get_cls_net(cfg)
    model_kp.load_state_dict(loaded_state)
    model_kp.to(device)
    model_kp.eval()

    loaded_state_l = torch.load(weights_line, map_location=device)
    model_line = get_cls_net_l(cfg_l)
    model_line.load_state_dict(loaded_state_l)
    model_line.to(device)
    model_line.eval()

    transform_resize = T.Resize((540, 960))
    print("Models loaded.")

    # ── Set up output video ─────────────────────────────────────────
    Config.OUTPUT_HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(Config.OUTPUT_HOMOGRAPHY_DIR / f"{args.match}_pnlcalib_{args.duration_sec}s.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # ── Load manual calibration seeds (if any) ──────────────────────
    manual_seeds = {}
    if not args.no_manual_seeds:
        manual_seeds = load_match_seeds(args.match)
        if manual_seeds:
            print(f"Loaded {len(manual_seeds)} manual keypoint seed(s) "
                  f"for {args.match}: frames {sorted(manual_seeds.keys())}")

    # ── Process frames ──────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cam = FramebyFrameCalib(iwidth=frame_w, iheight=frame_h, denormalize=True)
    smoother = ProjectionSmoother(
        alpha=args.alpha,
        max_stale_frames=int(fps * 3),  # reuse for up to 3 seconds
        reset_threshold=0.3,
        outlier_threshold=0.08,
        confirm_frames=3,
    )
    tracker = CameraMotionTracker() if not args.no_flow else None

    stats = {
        "success": 0, "sanity_fail": 0, "inference_fail": 0,
        "line_alignment_fail": 0, "non_gameplay": 0,
        "smoothed_reuse": 0, "outlier_rejected": 0, "no_projection": 0,
        "flow_propagated": 0, "flow_rejected_alignment": 0,
        "flow_streak_capped": 0, "manual_seed_used": 0,
    }
    flow_streak = 0  # consecutive flow-propagated frames since last PnLCalib anchor

    for i in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        abs_frame = start_frame + i

        # A) Non-gameplay filter: skip replays/close-ups
        if not is_gameplay_frame(frame, scoreboard_roi, scoreboard_threshold):
            stats["non_gameplay"] += 1
            # Non-gameplay disrupts feature tracking — drop the trail so
            # the next gameplay frame re-seeds cleanly.
            if tracker is not None:
                tracker.reset()
            flow_streak = 0
            out.write(frame)
            continue

        # B) Detect players for sanity check
        player_boxes = []
        for r in yolo(frame, verbose=False):
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    player_boxes.append((float(x1), float(y1), float(x2), float(y2)))

        # C) Run PnLCalib
        params = run_inference(cam, frame, model_kp, model_line, device, transform_resize)

        P_raw = None
        trusted = False
        if params is not None:
            P_raw = projection_from_cam_params(params)

            # Sanity check 1: do most players land on the pitch?
            if not check_projection_sanity(P_raw, player_boxes):
                P_raw = None
                stats["sanity_fail"] += 1
            # Sanity check 2: do projected lines match visible white pixels?
            elif not check_line_alignment(P_raw, frame, min_score=0.12):
                P_raw = None
                stats["line_alignment_fail"] += 1
            else:
                stats["success"] += 1
                trusted = True
        else:
            stats["inference_fail"] += 1

        # D) Fallback chain if PnLCalib didn't give us a trusted P:
        #    1) optical-flow propagation (capped by streak length + gated on
        #       player sanity AND line alignment, so drifted homographies
        #       don't silently ride along).
        #    2) nearest manual keypoint seed for this match (also gated).
        if tracker is not None:
            if trusted:
                tracker.seed(frame, P_raw)
                flow_streak = 0
            else:
                used_fallback = False

                if flow_streak >= args.max_flow_streak:
                    # Cap hit — tracker has been carrying too long without an
                    # absolute re-anchor. Drop the trail and wait for PnLCalib.
                    tracker.reset()
                    flow_streak = 0
                    stats["flow_streak_capped"] += 1
                else:
                    P_flow = tracker.propagate(frame)
                    if P_flow is not None:
                        ok_players = check_projection_sanity(P_flow, player_boxes)
                        ok_lines = ok_players and check_line_alignment(
                            P_flow, frame, min_score=0.12
                        )
                        if ok_lines:
                            P_raw = P_flow
                            stats["flow_propagated"] += 1
                            flow_streak += 1
                            used_fallback = True
                        else:
                            # Drifted — discard and drop the trail so next
                            # frame we either re-anchor via PnLCalib or a
                            # manual seed, not off the drifted trail.
                            tracker.reset()
                            flow_streak = 0
                            stats["flow_rejected_alignment"] += 1

                if not used_fallback and manual_seeds:
                    nearest_f = min(manual_seeds.keys(),
                                    key=lambda f: abs(f - abs_frame))
                    P_manual = manual_seeds[nearest_f]
                    if (check_projection_sanity(P_manual, player_boxes)
                            and check_line_alignment(P_manual, frame,
                                                     min_score=0.12)):
                        tracker.seed(frame, P_manual)
                        P_raw = P_manual
                        flow_streak = 0
                        stats["manual_seed_used"] += 1

        # E) Temporal smoothing with outlier rejection
        P_smooth = smoother.update(P_raw)

        if P_smooth is not None:
            if P_raw is None:
                stats["smoothed_reuse"] += 1
            frame = project_lines(frame, P_smooth, color=(0, 0, 255), thickness=2)
        else:
            stats["no_projection"] += 1

        out.write(frame)

    cap.release()
    out.release()

    total_processed = total_frames - stats["non_gameplay"]
    frames_with_lines = (
        stats["success"] + stats["smoothed_reuse"]
        + stats["flow_propagated"] + stats["manual_seed_used"]
    )
    print(f"\nDone! {total_frames} frames total ({stats['non_gameplay']} non-gameplay skipped):")
    print(f"  Gameplay frames processed: {total_processed}")
    print(f"  Frames WITH projection:    {frames_with_lines} ({100*frames_with_lines/max(total_processed,1):.1f}%)")
    print(f"    - PnLCalib success:      {stats['success']}")
    print(f"    - optical-flow carry:    {stats['flow_propagated']}")
    print(f"    - manual seed fallback:  {stats['manual_seed_used']}")
    print(f"    - stale (smoothed) hold: {stats['smoothed_reuse']}")
    print(f"  Frames WITHOUT projection: {stats['no_projection']}")
    print(f"    - PnLCalib sanity (players): {stats['sanity_fail']}")
    print(f"    - PnLCalib sanity (lines):   {stats['line_alignment_fail']}")
    print(f"    - PnLCalib inference failed: {stats['inference_fail']}")
    print(f"    - flow rejected (drifted):   {stats['flow_rejected_alignment']}")
    print(f"    - flow streak capped:        {stats['flow_streak_capped']}")
    print(f"  Smoother: {smoother.resets} resets, {smoother.outlier_rejected} outliers rejected")
    if tracker is not None:
        print(f"  Tracker: {tracker.seeds} seeds, {tracker.propagations} propagations, "
              f"{tracker.propagation_failures} failures")
    print(f"Output saved to: {output_path}")

    # Persist the stats dict so the notebook can build its coverage view
    # without us manually transcribing numbers after every run.
    coverage_pct = 100 * frames_with_lines / max(total_processed, 1)
    stats_path = Path(output_path).with_suffix(".stats.json")
    stats_payload = {
        "match":          args.match,
        "duration_sec":   args.duration_sec,
        "alpha":          args.alpha,
        "no_flow":        args.no_flow,
        "no_manual_seeds": args.no_manual_seeds,
        "max_flow_streak": args.max_flow_streak,
        "start_frame":    start_frame,
        "total_frames":   total_frames,
        "gameplay_frames": total_processed,
        "frames_with_lines": frames_with_lines,
        "coverage_pct":   coverage_pct,
        "stats":          stats,
        "smoother": {
            "resets": smoother.resets,
            "outlier_rejected": smoother.outlier_rejected,
        },
        "tracker": (
            {
                "seeds": tracker.seeds,
                "propagations": tracker.propagations,
                "propagation_failures": tracker.propagation_failures,
            }
            if tracker is not None else None
        ),
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2))
    print(f"Stats saved to:  {stats_path}")


if __name__ == "__main__":
    main()
