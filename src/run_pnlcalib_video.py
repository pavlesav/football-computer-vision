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


def check_line_alignment(P, frame, min_score=0.05):
    """
    Verify that projected pitch lines roughly align with bright line pixels
    in the image. Projects a sparse set of world-space pitch line points,
    checks how many land near bright pixels in a generous neighborhood.

    This catches projections where the player sanity check passes by
    coincidence but the lines are clearly in the wrong place (e.g.,
    projected onto stands or buildings).

    Deliberately lenient — we want to catch gross errors, not fine-tune.
    """
    h, w = frame.shape[:2]
    # Build a bright-line mask: high brightness, not too saturated
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 140)

    # Large dilation — we're checking rough alignment, not pixel-perfect
    kernel = np.ones((11, 11), np.uint8)
    white_mask = cv2.dilate(white_mask.astype(np.uint8), kernel, iterations=2)

    # Sample points along a few key pitch lines (not all — many are off-screen)
    # Use only the most visible lines: touchlines, halfway line, PA edges
    key_lines = [
        [[0., 68., 0.], [105., 68., 0.]],     # near touchline
        [[0., 0., 0.], [105., 0., 0.]],        # far touchline
        [[52.5, 0., 0.], [52.5, 68, 0.]],      # halfway line
        [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], # left PA side
        [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], # right PA side
        [[0., 54.16, 0.], [16.5, 54.16, 0.]],   # left PA top
        [[88.5, 54.16, 0.], [105., 54.16, 0.]], # right PA top
    ]

    hits = 0
    total = 0
    for line in key_lines:
        length = np.sqrt(sum((a - b) ** 2 for a, b in zip(line[0], line[1])))
        n = max(3, int(length / 8))  # sparse sampling
        for t in np.linspace(0, 1, n):
            wx = line[0][0] + t * (line[1][0] - line[0][0])
            wy = line[0][1] + t * (line[1][1] - line[0][1])
            ip = _project_point(P, (wx, wy, 0.0))
            if ip is None:
                continue
            ix, iy = int(round(ip[0])), int(round(ip[1]))
            if 0 <= ix < w and 0 <= iy < h:
                total += 1
                if white_mask[iy, ix]:
                    hits += 1

    if total < 5:
        return False
    return (hits / total) >= min_score


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
                 reset_threshold=0.3, outlier_threshold=0.08,
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
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

    return final_params_dict


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
        "flow_propagated": 0, "manual_seed_used": 0,
    }

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
        #    1) optical-flow propagation from the last good projection
        #    2) nearest manual keypoint seed for this match (if any)
        if tracker is not None:
            if trusted:
                tracker.seed(frame, P_raw)
            else:
                P_flow = tracker.propagate(frame)
                if P_flow is not None and check_projection_sanity(P_flow, player_boxes):
                    P_raw = P_flow
                    stats["flow_propagated"] += 1
                elif manual_seeds:
                    # Use nearest manual seed as a last resort, then let the
                    # tracker propagate from it for the upcoming frames.
                    nearest_f = min(manual_seeds.keys(), key=lambda f: abs(f - abs_frame))
                    P_manual = manual_seeds[nearest_f]
                    if check_projection_sanity(P_manual, player_boxes):
                        P_raw = tracker.seed(frame, P_manual)
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
    print(f"    - sanity check (players): {stats['sanity_fail']}")
    print(f"    - sanity check (lines):   {stats['line_alignment_fail']}")
    print(f"    - inference failed:       {stats['inference_fail']}")
    print(f"  Smoother: {smoother.resets} resets, {smoother.outlier_rejected} outliers rejected")
    if tracker is not None:
        print(f"  Tracker: {tracker.seeds} seeds, {tracker.propagations} propagations, "
              f"{tracker.propagation_failures} failures")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
