"""
Pitch homography estimation from broadcast football footage.

Estimates the 3x3 homography matrix that maps image pixels to pitch
coordinates (meters on a 105x68m FIFA-standard pitch). Works on the
standard side-on elevated broadcast camera angle used in 1.CFL.

Pipeline:
  1. Field segmentation (HSV green mask)
  2. White line detection with player suppression
  3. Line segment detection + merging
  4. Semantic line classification (touchlines vs perpendicular lines)
  5. Keypoint extraction from classified line intersections
  6. Keypoint-to-template matching
  7. Homography estimation via RANSAC
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ── FIFA pitch template ──────────────────────────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
CENTER_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DIST = 11.0

# Named keypoints: (x, y) in meters, origin bottom-left of pitch
PITCH_KEYPOINTS = {
    # Pitch corners
    "BL": (0, 0), "BR": (PITCH_LENGTH, 0),
    "TL": (0, PITCH_WIDTH), "TR": (PITCH_LENGTH, PITCH_WIDTH),
    # Center line meets touchlines
    "CB": (PITCH_LENGTH / 2, 0), "CT": (PITCH_LENGTH / 2, PITCH_WIDTH),
    "CC": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    # Left penalty area
    "LPA_B": (PENALTY_AREA_DEPTH, (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2),
    "LPA_T": (PENALTY_AREA_DEPTH, (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2),
    "L_PA_GL_B": (0, (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2),
    "L_PA_GL_T": (0, (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2),
    # Right penalty area
    "RPA_B": (PITCH_LENGTH - PENALTY_AREA_DEPTH, (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2),
    "RPA_T": (PITCH_LENGTH - PENALTY_AREA_DEPTH, (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2),
    "R_PA_GL_B": (PITCH_LENGTH, (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2),
    "R_PA_GL_T": (PITCH_LENGTH, (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2),
    # Left goal area
    "LGA_B": (GOAL_AREA_DEPTH, (PITCH_WIDTH - GOAL_AREA_WIDTH) / 2),
    "LGA_T": (GOAL_AREA_DEPTH, (PITCH_WIDTH + GOAL_AREA_WIDTH) / 2),
    # Right goal area
    "RGA_B": (PITCH_LENGTH - GOAL_AREA_DEPTH, (PITCH_WIDTH - GOAL_AREA_WIDTH) / 2),
    "RGA_T": (PITCH_LENGTH - GOAL_AREA_DEPTH, (PITCH_WIDTH + GOAL_AREA_WIDTH) / 2),
    # Center circle meets halfway line
    "CC_B": (PITCH_LENGTH / 2, PITCH_WIDTH / 2 - CENTER_CIRCLE_RADIUS),
    "CC_T": (PITCH_LENGTH / 2, PITCH_WIDTH / 2 + CENTER_CIRCLE_RADIUS),
}

# Pitch lines as (name, start_keypoint, end_keypoint)
PITCH_LINES = {
    # Touchlines (run along the length)
    "near_touchline": ("BL", "BR"),
    "far_touchline": ("TL", "TR"),
    # Goal lines
    "left_goal_line": ("BL", "TL"),
    "right_goal_line": ("BR", "TR"),
    # Halfway line
    "halfway_line": ("CB", "CT"),
    # Left penalty area
    "left_pa_bottom": ("L_PA_GL_B", "LPA_B"),
    "left_pa_top": ("L_PA_GL_T", "LPA_T"),
    "left_pa_edge": ("LPA_B", "LPA_T"),
    # Right penalty area
    "right_pa_bottom": ("R_PA_GL_B", "RPA_B"),
    "right_pa_top": ("R_PA_GL_T", "RPA_T"),
    "right_pa_edge": ("RPA_B", "RPA_T"),
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class DetectedLine:
    """A detected line segment with semantic label."""
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float       # degrees, 0-180
    label: str = ""    # semantic label: "touchline", "perpendicular", ""

    @property
    def length(self):
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def segment(self):
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class HomographyResult:
    """Result of homography estimation for one frame."""
    H: Optional[np.ndarray]        # 3x3 homography matrix (image → pitch)
    n_inliers: int = 0
    n_matches: int = 0
    reprojection_error: float = float('inf')
    image_points: Optional[np.ndarray] = None
    pitch_points: Optional[np.ndarray] = None
    point_labels: Optional[List[str]] = None
    lines: Optional[List[DetectedLine]] = None


# ── Field segmentation ───────────────────────────────────────────────

def detect_field_mask(frame: np.ndarray) -> np.ndarray:
    """Segment the green pitch area. Returns binary mask (255=field)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([90, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    return mask


# ── White line detection ─────────────────────────────────────────────

def detect_line_pixels(
    frame: np.ndarray,
    field_mask: np.ndarray,
    player_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """
    Detect white pitch line pixels within the field area.

    Uses two complementary detection channels:
      1. Color-based: bright + low saturation pixels (catches clean white lines)
      2. Edge-based: Canny edges in bright regions (catches worn/dim lines)

    Also masks out player bounding boxes and filters small/compact blobs.

    Args:
        frame: BGR broadcast frame.
        field_mask: Binary mask of the pitch area.
        player_boxes: Optional list of (x1, y1, x2, y2) person bounding boxes
                      to mask out before line detection.

    Returns:
        Binary mask of detected white line pixels.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Erode field mask to avoid boundary artifacts
    work_mask = cv2.erode(field_mask, np.ones((15, 15), np.uint8))
    # Mask broadcast overlays
    work_mask[:int(h * 0.13), :] = 0           # scoreboard
    work_mask[int(h * 0.87):, int(w * 0.75):] = 0  # logo
    work_mask[int(h * 0.87):, :int(w * 0.08)] = 0

    # Mask out detected players (biggest source of false white lines)
    if player_boxes:
        for bx1, by1, bx2, by2 in player_boxes:
            pad = 10
            bx1 = max(0, bx1 - pad)
            by1 = max(0, by1 - pad)
            bx2 = min(w, bx2 + pad)
            by2 = min(h, by2 + pad)
            work_mask[by1:by2, bx1:bx2] = 0

    # Compute field brightness stats for adaptive thresholds
    field_v = hsv[:, :, 2][work_mask > 0]
    if len(field_v) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    med_v = float(np.median(field_v))
    std_v = float(np.std(field_v))

    # --- Channel 1: Color-based white detection ---
    # Bright + low saturation = white lines
    val_min = max(int(med_v + 1.5 * std_v), 130)
    color_mask = cv2.inRange(hsv, np.array([0, 0, val_min]), np.array([180, 80, 255]))
    color_mask = cv2.bitwise_and(color_mask, work_mask)

    # --- Channel 2: White top-hat transform ---
    # CLAHE boosts local contrast — use moderate clipLimit to avoid
    # amplifying grass texture on dark fields
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Top-hat highlights thin bright structures (lines) on darker background
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, tophat_kernel)
    tophat_mask = cv2.bitwise_and(tophat, work_mask)
    # Blur to suppress speckle noise before thresholding
    tophat_mask = cv2.GaussianBlur(tophat_mask, (5, 5), 0)
    # Adaptive threshold: lower for dark fields
    if med_v < 100:
        tophat_thresh = max(int(std_v * 0.4), 18)
    else:
        tophat_thresh = max(int(std_v * 0.5), 25)
    _, tophat_bin = cv2.threshold(tophat_mask, tophat_thresh, 255, cv2.THRESH_BINARY)

    # Combine both channels
    combined = cv2.bitwise_or(color_mask, tophat_bin)

    # Morphological cleanup
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, thin_kernel)

    # Connected component filtering: keep elongated shapes only
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    filtered = np.zeros_like(combined)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        cw = stats[lbl, cv2.CC_STAT_WIDTH]
        ch = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(cw, ch) / (min(cw, ch) + 1e-5)

        if area < 25:
            continue
        # Reject compact blobs (players, balls) — keep elongated shapes
        if aspect < 2.0 and area > 300:
            continue
        filtered[labels == lbl] = 255

    return filtered


# ── Line segment detection ───────────────────────────────────────────

def detect_lines(
    line_mask: np.ndarray,
    min_length: int = 60,
    max_gap: int = 20,
) -> List[DetectedLine]:
    """Detect line segments using Probabilistic Hough Transform."""
    raw = cv2.HoughLinesP(
        line_mask, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=min_length, maxLineGap=max_gap,
    )
    if raw is None:
        return []

    lines = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        lines.append(DetectedLine(x1, y1, x2, y2, angle))
    return lines


def validate_line_support(line: DetectedLine, mask: np.ndarray, min_frac: float = 0.30) -> bool:
    """Check that enough pixels along the line lie on the white mask."""
    n = max(int(line.length), 20)
    xs = np.linspace(line.x1, line.x2, n).astype(int)
    ys = np.linspace(line.y1, line.y2, n).astype(int)
    h, w = mask.shape[:2]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        return False
    return float(np.mean(mask[ys, xs] > 0)) >= min_frac


def merge_lines(
    lines: List[DetectedLine],
    line_mask: np.ndarray,
    angle_thresh: float = 10.0,
    dist_thresh: float = 50.0,
    min_merged_length: int = 150,
) -> List[DetectedLine]:
    """Merge co-linear line segments into single representatives."""
    # Pre-filter: only keep lines with real white-pixel support
    lines = [l for l in lines if validate_line_support(l, line_mask)]
    if not lines:
        return []

    # Sort by length (longest first)
    lines.sort(key=lambda l: -l.length)

    used = [False] * len(lines)
    merged = []

    for i, li in enumerate(lines):
        if used[i]:
            continue
        used[i] = True
        cluster = [li]

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            lj = lines[j]

            # Angle similarity
            ad = min(abs(li.angle - lj.angle), 180 - abs(li.angle - lj.angle))
            if ad > angle_thresh:
                continue

            # Perpendicular distance between midpoints
            rad = np.radians(li.angle)
            dx = lj.midpoint[0] - li.midpoint[0]
            dy = lj.midpoint[1] - li.midpoint[1]
            perp = abs(dx * np.sin(rad) - dy * np.cos(rad))
            if perp < dist_thresh:
                cluster.append(lj)
                used[j] = True

        # Fit a line through all endpoints of the cluster
        pts = np.array([[l.x1, l.y1, l.x2, l.y2] for l in cluster]).reshape(-1, 2)
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        proj = (pts[:, 0] - cx) * vx + (pts[:, 1] - cy) * vy
        t_min, t_max = proj.min(), proj.max()
        angle = np.degrees(np.arctan2(vy, vx)) % 180
        merged.append(DetectedLine(
            int(cx + vx * t_min), int(cy + vy * t_min),
            int(cx + vx * t_max), int(cy + vy * t_max),
            angle,
        ))

    # Post-merge filter: remove short lines
    merged = [l for l in merged if l.length >= min_merged_length]
    return merged


# ── Semantic line classification ─────────────────────────────────────

def classify_lines(
    lines: List[DetectedLine],
    frame_shape: Tuple[int, int],
    horizontal_angle_range: float = 15.0,
) -> List[DetectedLine]:
    """
    Classify merged lines as 'touchline' (horizontal) or 'perpendicular'.

    In a standard broadcast view:
    - Touchlines are roughly horizontal (angle within ~15° of horizontal)
    - Perpendicular lines (halfway, PA edges, goal lines) converge toward
      a vanishing point and appear at steeper angles (typically 50-90°)
    - Lines between 15-50° are ambiguous (could be penalty area top/bottom
      in heavy perspective) — classified as 'other'
    """
    h, w = frame_shape[:2]

    for line in lines:
        a = line.angle
        horiz_dev = min(a, 180 - a)  # deviation from horizontal

        if horiz_dev < horizontal_angle_range:
            line.label = "touchline"
        elif horiz_dev > 45:
            line.label = "perpendicular"
        else:
            # 15-45°: likely noise from center circle / penalty arcs
            line.label = "other"

    return lines


def identify_touchlines(
    lines: List[DetectedLine],
    frame_shape: Tuple[int, int],
    min_length_frac: float = 0.15,
) -> Tuple[Optional[DetectedLine], Optional[DetectedLine]]:
    """
    Identify the near (bottom) and far (top) touchlines.

    Returns (near_touchline, far_touchline). Either can be None.
    """
    h, w = frame_shape[:2]
    min_len = w * min_length_frac

    touchline_candidates = [
        l for l in lines
        if l.label == "touchline" and l.length >= min_len
    ]

    if not touchline_candidates:
        return None, None

    # Sort by average y-position (bottom of frame = high y)
    touchline_candidates.sort(key=lambda l: (l.y1 + l.y2) / 2)

    near = touchline_candidates[-1]  # highest y = bottom of frame = near touchline

    if len(touchline_candidates) >= 2:
        far = touchline_candidates[0]   # lowest y = top = far touchline
        # Check they're reasonably separated (at least 15% of frame height)
        near_y = (near.y1 + near.y2) / 2
        far_y = (far.y1 + far.y2) / 2
        if near_y - far_y < h * 0.10:
            far = None
    else:
        far = None

    return near, far


# ── Perpendicular line validation ────────────────────────────────────

def _validate_perpendicular_line(
    line: DetectedLine,
    near_tl: DetectedLine,
    far_tl: DetectedLine,
    frame_shape: Tuple[int, int],
) -> bool:
    """
    Check that a perpendicular line is geometrically consistent with a real
    pitch line: it must extend from one touchline to the other.

    A real perpendicular line (halfway, PA edge, goal line) runs the full
    width of the pitch. In the image, it should intersect both touchlines
    at reasonable x-positions and its detected segment should lie between
    the touchlines.  Circle/arc fragments fail this check because they're
    short and don't span the full field height.
    """
    h, w = frame_shape[:2]
    margin_x = w * 0.3  # allow intersection slightly off-screen

    # Must intersect both touchlines
    near_pt = _line_intersection(line, near_tl)
    far_pt = _line_intersection(line, far_tl)
    if near_pt is None or far_pt is None:
        return False

    # Intersections must be within frame width (with margin)
    if not (-margin_x <= near_pt[0] <= w + margin_x):
        return False
    if not (-margin_x <= far_pt[0] <= w + margin_x):
        return False

    # The detected segment must lie between (or near) the two touchlines
    near_y = (near_tl.y1 + near_tl.y2) / 2
    far_y = (far_tl.y1 + far_tl.y2) / 2
    tl_gap = abs(near_y - far_y)
    y_min = min(near_y, far_y) - tl_gap * 0.15
    y_max = max(near_y, far_y) + tl_gap * 0.15

    # At least one endpoint must be within the touchline band
    line_y_min = min(line.y1, line.y2)
    line_y_max = max(line.y1, line.y2)
    if line_y_max < y_min or line_y_min > y_max:
        return False

    # The detected segment should span at least 15% of the touchline gap
    overlap_top = max(line_y_min, y_min)
    overlap_bot = min(line_y_max, y_max)
    if (overlap_bot - overlap_top) < tl_gap * 0.15:
        return False

    return True


# ── Intersection detection ───────────────────────────────────────────

def _line_intersection(seg1: DetectedLine, seg2: DetectedLine) -> Optional[Tuple[float, float]]:
    """Intersection of two infinite lines through the segments."""
    x1, y1, x2, y2 = seg1.x1, seg1.y1, seg1.x2, seg1.y2
    x3, y3, x4, y4 = seg2.x1, seg2.y1, seg2.x2, seg2.y2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def find_keypoints(
    lines: List[DetectedLine],
    frame_shape: Tuple[int, int],
    min_angle: float = 15.0,
) -> List[Tuple[float, float, str, str]]:
    """
    Find intersections between classified lines.

    Returns list of (x, y, line1_label, line2_label) tuples.
    Only keeps intersections within the frame between lines of different classes.
    """
    h, w = frame_shape[:2]
    margin = 100
    keypoints = []

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            li, lj = lines[i], lines[j]

            # Only intersect lines that aren't in the same family
            if li.label == lj.label and li.label == "touchline":
                continue
            if li.label == lj.label and li.label == "perpendicular":
                continue
            if not li.label or not lj.label:
                continue

            # Check angle difference
            ad = min(abs(li.angle - lj.angle), 180 - abs(li.angle - lj.angle))
            if ad < min_angle:
                continue

            pt = _line_intersection(li, lj)
            if pt is None:
                continue

            x, y = pt
            if -margin <= x <= w + margin and -margin <= y <= h + margin:
                keypoints.append((x, y, li.label, lj.label))

    return keypoints


# ── Keypoint-to-template matching ────────────────────────────────────

# Known x-coordinates of perpendicular pitch lines (meters from left goal line)
PITCH_LINE_X = {
    "left_goal_line": 0.0,
    "left_goal_area": GOAL_AREA_DEPTH,          # 5.5
    "left_pa_edge": PENALTY_AREA_DEPTH,          # 16.5
    "halfway": PITCH_LENGTH / 2,                 # 52.5
    "right_pa_edge": PITCH_LENGTH - PENALTY_AREA_DEPTH,  # 88.5
    "right_goal_area": PITCH_LENGTH - GOAL_AREA_DEPTH,   # 99.5
    "right_goal_line": PITCH_LENGTH,             # 105.0
}


def _identify_perpendicular_lines(
    perp_lines: List[DetectedLine],
    near_tl: DetectedLine,
    far_tl: DetectedLine,
    frame_shape: Tuple[int, int],
) -> List[Tuple[DetectedLine, float, str]]:
    """
    Identify which pitch line each perpendicular corresponds to.

    Returns list of (line, pitch_x, label) for successfully identified lines.

    Strategy:
    1. Compute x-intercept of each perpendicular with the near touchline
    2. Sort by x-position (left to right in image = direction on pitch)
    3. Try to match spacing patterns to known pitch geometry
    """
    h, w = frame_shape[:2]

    # Get x-intercept of each perp with near touchline
    intercepts = []
    for pline in perp_lines:
        pt = _line_intersection(pline, near_tl)
        if pt is None:
            continue
        x, y = pt
        if -100 <= x <= w + 100:
            intercepts.append((x, pline))

    if len(intercepts) < 1:
        return []

    # Sort left to right by x-intercept
    intercepts.sort(key=lambda t: t[0])

    # Single perpendicular line: identify by position heuristic
    if len(intercepts) == 1:
        return _match_single_perp(intercepts[0], near_tl, far_tl, frame_shape)

    # Compute pairwise pixel distances between perpendicular lines
    xs = [x for x, _ in intercepts]

    # Also compute the "scale" from the touchlines: the near touchline is wider
    # (closer to camera) than the far touchline. The ratio gives perspective info.
    near_len = near_tl.length
    far_len = far_tl.length

    # Estimate pixels-per-meter at the near touchline level.
    # The visible pitch width is covered by the near touchline length.
    # We don't know exactly how many meters the near touchline covers,
    # but we can use the perpendicular line spacing to figure it out.

    # Try matching the spacing pattern to known pitch line positions.
    # Known spacings (in meters):
    #   halfway - left_pa = 36.0
    #   halfway - right_pa = 36.0
    #   left_pa - left_ga = 11.0
    #   right_pa - right_ga = 11.0
    #   left_ga - left_goal = 5.5
    #   right_ga - right_goal = 5.5
    #   left_pa - left_goal = 16.5
    #   right_pa - right_goal = 16.5

    # For 2 perpendicular lines, try each possible pair assignment
    # and check consistency with the visible field extent.
    if len(intercepts) == 2:
        return _match_two_perps(intercepts, near_tl, far_tl, frame_shape)

    # For 3+ perpendicular lines, use pairwise spacing ratios
    return _match_multiple_perps(intercepts, near_tl, far_tl, frame_shape)


def _match_single_perp(
    intercept: Tuple[float, DetectedLine],
    near_tl: DetectedLine,
    far_tl: DetectedLine,
    frame_shape: Tuple[int, int],
) -> List[Tuple[DetectedLine, float, str]]:
    """
    Identify a single perpendicular line by its position in the frame.

    Heuristic: the camera is typically near the halfway line, so:
    - Perpendicular near frame center → halfway line (x=52.5)
    - Perpendicular in left third → left PA edge (x=16.5)
    - Perpendicular in right third → right PA edge (x=88.5)
    """
    h, w = frame_shape[:2]
    x, pline = intercept

    # Normalize position: 0=left edge, 1=right edge
    frac = x / w

    if 0.35 <= frac <= 0.65:
        # Near center → halfway line
        return [(pline, PITCH_LENGTH / 2, "halfway")]
    elif frac < 0.35:
        # Left side → left PA edge
        return [(pline, PENALTY_AREA_DEPTH, "left_pa_edge")]
    else:
        # Right side → right PA edge
        return [(pline, PITCH_LENGTH - PENALTY_AREA_DEPTH, "right_pa_edge")]


def _match_two_perps(
    intercepts: List[Tuple[float, DetectedLine]],
    near_tl: DetectedLine,
    far_tl: DetectedLine,
    frame_shape: Tuple[int, int],
) -> List[Tuple[DetectedLine, float, str]]:
    """Match exactly 2 perpendicular lines to pitch line identities."""
    h, w = frame_shape[:2]
    x_left, line_left = intercepts[0]
    x_right, line_right = intercepts[1]
    pixel_gap = x_right - x_left
    if pixel_gap < 20:
        return []

    # The center of the two lines in the image
    center_x = (x_left + x_right) / 2

    # Candidate assignments: (left_line_name, right_line_name, gap_meters)
    candidates = [
        ("left_pa_edge", "halfway", 36.0),
        ("halfway", "right_pa_edge", 36.0),
        ("left_goal_area", "left_pa_edge", 11.0),
        ("right_pa_edge", "right_goal_area", 11.0),
        ("left_goal_line", "left_pa_edge", 16.5),
        ("right_pa_edge", "right_goal_line", 16.5),
        ("left_goal_line", "left_goal_area", 5.5),
        ("right_goal_area", "right_goal_line", 5.5),
        ("left_pa_edge", "right_pa_edge", 72.0),
    ]

    # For each candidate, compute the implied pixels_per_meter
    # and check if it's consistent with the frame geometry
    best_score = -1
    best_assignment = None

    for left_name, right_name, gap_m in candidates:
        ppm = pixel_gap / gap_m  # pixels per meter at near touchline

        # The near touchline covers near_tl.length pixels.
        # That corresponds to near_tl.length / ppm meters.
        visible_m = near_tl.length / ppm

        # Sanity: visible portion should be between 30m and 105m
        if visible_m < 25 or visible_m > 110:
            continue

        # The implied pitch x of the left edge of the near touchline
        left_x_pitch = PITCH_LINE_X[left_name] - (x_left - near_tl.x1) / ppm
        # Should be between -5 and 80 (allowing some margin)
        if left_x_pitch < -10 or left_x_pitch > 85:
            continue

        right_x_pitch = left_x_pitch + visible_m
        if right_x_pitch < 20 or right_x_pitch > 115:
            continue

        # Score: prefer assignments where the halfway line is closer
        # to the center of the frame (camera usually centered there)
        halfway_pixel = near_tl.x1 + (52.5 - left_x_pitch) * ppm
        dist_from_center = abs(halfway_pixel - w / 2) / w

        # Also prefer assignments that give reasonable ppm (not too high/low)
        ppm_score = 1.0 - abs(ppm - 15.0) / 30.0  # expect ~10-20 ppm
        ppm_score = max(0, ppm_score)

        score = (1.0 - dist_from_center) * 0.6 + ppm_score * 0.4

        if score > best_score:
            best_score = score
            best_assignment = (left_name, right_name)

    if best_assignment is None:
        return []

    left_name, right_name = best_assignment
    return [
        (line_left, PITCH_LINE_X[left_name], left_name),
        (line_right, PITCH_LINE_X[right_name], right_name),
    ]


def _match_multiple_perps(
    intercepts: List[Tuple[float, DetectedLine]],
    near_tl: DetectedLine,
    far_tl: DetectedLine,
    frame_shape: Tuple[int, int],
) -> List[Tuple[DetectedLine, float, str]]:
    """Match 3+ perpendicular lines using spacing ratios."""
    # Use the two widest-spaced lines as anchors, then verify others
    xs = [x for x, _ in intercepts]
    n = len(xs)

    # Try all pairs as anchor pair
    best_score = -1
    best_result = []

    for i in range(n):
        for j in range(i + 1, n):
            anchor_left = intercepts[i]
            anchor_right = intercepts[j]

            result = _match_two_perps(
                [anchor_left, anchor_right], near_tl, far_tl,
                frame_shape,
            )
            if not result:
                continue

            _, pitch_x_left, name_left = result[0]
            _, pitch_x_right, name_right = result[1]
            ppm = (anchor_right[0] - anchor_left[0]) / (pitch_x_right - pitch_x_left)

            # Score: how many other perp lines match known pitch lines?
            total_matched = 2
            full_result = list(result)

            for k in range(n):
                if k == i or k == j:
                    continue
                x_k = intercepts[k][0]
                implied_pitch_x = pitch_x_left + (x_k - anchor_left[0]) / ppm

                # Check if this pitch_x is close to any known line
                for lname, lx in PITCH_LINE_X.items():
                    if lname in (name_left, name_right):
                        continue
                    if abs(implied_pitch_x - lx) < 3.0:  # within 3m tolerance
                        full_result.append(
                            (intercepts[k][1], lx, lname)
                        )
                        total_matched += 1
                        break

            if total_matched > best_score:
                best_score = total_matched
                best_result = full_result

    return best_result


# ── Vanishing point filtering ────────────────────────────────────────

def _filter_by_vanishing_point(
    perps: List[DetectedLine],
    max_dist: float = 80.0,
) -> List[DetectedLine]:
    """
    Filter perpendicular lines by vanishing-point consensus.

    All real perpendicular pitch lines are parallel in 3D space, so they
    converge to a single vanishing point in the image. Lines from circles
    or arcs point in different directions and won't converge to the same VP.

    Uses RANSAC-style approach: find the VP supported by the most lines,
    then keep only lines that pass near it.
    """
    if len(perps) < 3:
        return perps

    # Compute all pairwise intersections (VP candidates)
    vp_candidates = []
    for i in range(len(perps)):
        for j in range(i + 1, len(perps)):
            pt = _line_intersection(perps[i], perps[j])
            if pt is not None:
                vp_candidates.append((pt, i, j))

    if not vp_candidates:
        return perps

    # For each VP candidate, count how many lines pass close to it
    best_inliers = []
    for (vx, vy), _, _ in vp_candidates:
        inliers = []
        for k, line in enumerate(perps):
            dx = line.x2 - line.x1
            dy = line.y2 - line.y1
            length = np.sqrt(dx**2 + dy**2)
            if length < 1:
                continue
            # Point-to-line distance from VP to the extended line
            dist = abs(dx * (line.y1 - vy) - dy * (line.x1 - vx)) / length
            if dist < max_dist:
                inliers.append(k)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) >= 2:
        return [perps[i] for i in best_inliers]
    return perps


# ── Full pipeline ────────────────────────────────────────────────────

def estimate_homography(
    frame: np.ndarray,
    player_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> HomographyResult:
    """
    Estimate pitch homography from a single broadcast frame.

    Args:
        frame: BGR broadcast frame (1920x1080).
        player_boxes: YOLO-detected person bounding boxes to mask out.

    Returns:
        HomographyResult with homography matrix and diagnostics.
    """
    h, w = frame.shape[:2]

    # Early reject: close-ups and replays have few visible players.
    # A game-play wide shot should have at least 8 detected persons.
    if player_boxes is not None and len(player_boxes) < 8:
        return HomographyResult(H=None)

    # Step 1: Field segmentation
    field_mask = detect_field_mask(frame)

    # Step 2: White line detection (with player suppression)
    line_mask = detect_line_pixels(frame, field_mask, player_boxes)

    # Step 3: Line segment detection + merging
    raw_lines = detect_lines(line_mask)
    lines = merge_lines(raw_lines, line_mask)

    # Step 4: Classify lines
    lines = classify_lines(lines, (h, w))

    # Step 5: Identify touchlines
    near_tl, far_tl = identify_touchlines(lines, (h, w))

    if near_tl is None or far_tl is None:
        return HomographyResult(H=None, lines=lines)

    # Step 6: Get perpendicular lines (exclude "other" — those are mostly
    # circle/arc fragments at ambiguous angles, not real pitch lines)
    perps = [l for l in lines if l.label == "perpendicular" and l.length >= w * 0.04]

    # Validate: each perpendicular must span between both touchlines
    perps = [l for l in perps
             if _validate_perpendicular_line(l, near_tl, far_tl, (h, w))]

    # Vanishing point consensus: all real perpendicular pitch lines are
    # parallel in 3D → they converge to one VP in the image. Reject any
    # line that doesn't pass near the consensus VP.
    if len(perps) >= 3:
        perps = _filter_by_vanishing_point(perps)

    # Cap at 5 — a single broadcast view never shows more than 5
    # perpendicular pitch lines. Keep the longest (most reliable).
    if len(perps) > 5:
        perps.sort(key=lambda l: -l.length)
        perps = perps[:5]

    if len(perps) < 1:
        return HomographyResult(H=None, lines=lines,
                                n_matches=0)

    # Step 7: Identify which pitch lines the perpendiculars are
    identified = _identify_perpendicular_lines(perps, near_tl, far_tl, (h, w))
    if len(identified) < 1:
        return HomographyResult(H=None, lines=lines, n_matches=0)

    # Step 8: Build image↔pitch point correspondences
    # Each identified perp × each touchline = one correspondence
    image_points = []
    pitch_points = []
    point_labels = []

    for pline, pitch_x, pname in identified:
        # Intersection with near touchline (y=0 in pitch coords)
        near_pt = _line_intersection(pline, near_tl)
        if near_pt is not None:
            ix, iy = near_pt
            if -100 <= ix <= w + 100 and -100 <= iy <= h + 100:
                image_points.append([ix, iy])
                pitch_points.append([pitch_x, 0.0])
                point_labels.append(f"{pname}×near")

        # Intersection with far touchline (y=68 in pitch coords)
        far_pt = _line_intersection(pline, far_tl)
        if far_pt is not None:
            ix, iy = far_pt
            if -100 <= ix <= w + 100 and -100 <= iy <= h + 100:
                image_points.append([ix, iy])
                pitch_points.append([pitch_x, PITCH_WIDTH])
                point_labels.append(f"{pname}×far")

    # If we only have 2 points (1 perpendicular × 2 touchlines), add
    # touchline endpoints to get the minimum 4 points for homography.
    if 2 <= len(image_points) < 4 and len(identified) >= 1:
        # Estimate pixels-per-meter from the identified perpendicular
        pline, pitch_x, pname = identified[0]
        near_pt = _line_intersection(pline, near_tl)
        far_pt = _line_intersection(pline, far_tl)

        if near_pt is not None and far_pt is not None:
            # Use perspective ratio between near and far touchline lengths
            # to estimate ppm at each level
            near_len = near_tl.length
            far_len = far_tl.length

            # Heuristic: typical broadcast shows ~50-70m of pitch width
            # Use the ratio of touchline lengths to estimate visible range
            # Camera roughly centered → the perpendicular at pitch_x divides
            # the visible area proportionally
            near_x = near_pt[0]
            # Fraction of near touchline to the left of the perpendicular
            frac_left = (near_x - near_tl.x1) / (near_tl.x2 - near_tl.x1 + 1e-5)
            frac_right = 1.0 - frac_left

            # Estimate visible meters assuming ~15 ppm at near touchline
            # and use the perpendicular as anchor
            ppm_est = near_len / 60.0  # assume ~60m visible
            pitch_left = pitch_x - frac_left * near_len / ppm_est
            pitch_right = pitch_x + frac_right * near_len / ppm_est

            # Clamp to pitch bounds
            pitch_left = max(0, min(pitch_left, PITCH_LENGTH - 10))
            pitch_right = max(10, min(pitch_right, PITCH_LENGTH))

            # Add near touchline endpoints
            image_points.append([near_tl.x1, near_tl.y1])
            pitch_points.append([pitch_left, 0.0])
            point_labels.append("near_tl_left")

            image_points.append([near_tl.x2, near_tl.y2])
            pitch_points.append([pitch_right, 0.0])
            point_labels.append("near_tl_right")

            # Add far touchline endpoints
            far_x = far_pt[0]
            frac_left_f = (far_x - far_tl.x1) / (far_tl.x2 - far_tl.x1 + 1e-5)
            frac_right_f = 1.0 - frac_left_f
            pitch_left_f = pitch_x - frac_left_f * far_len / (far_len / (pitch_right - pitch_left) * near_len / near_len + 1e-5)

            # Simpler: use same pitch_left/right for far (the perspective
            # transform handles the scaling difference)
            image_points.append([far_tl.x1, far_tl.y1])
            pitch_points.append([pitch_left, PITCH_WIDTH])
            point_labels.append("far_tl_left")

            image_points.append([far_tl.x2, far_tl.y2])
            pitch_points.append([pitch_right, PITCH_WIDTH])
            point_labels.append("far_tl_right")

    if len(image_points) < 4:
        return HomographyResult(H=None, lines=lines,
                                n_matches=len(image_points))

    img_pts = np.array(image_points, dtype=np.float64)
    pit_pts = np.array(pitch_points, dtype=np.float64)

    # Step 9: Compute homography
    # Use 5.0m RANSAC threshold — touchline endpoint coordinates are estimated,
    # so strict thresholds reject valid correspondences
    H, mask = cv2.findHomography(img_pts, pit_pts, cv2.RANSAC, 5.0)

    if H is None:
        return HomographyResult(H=None, lines=lines,
                                n_matches=len(image_points))

    n_inliers = int(mask.sum()) if mask is not None else 0

    # Compute reprojection error (inliers only)
    projected = cv2.perspectiveTransform(
        img_pts.reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    if mask is not None:
        inlier_mask = mask.flatten().astype(bool)
        if inlier_mask.any():
            errors = np.sqrt(np.sum(
                (projected[inlier_mask] - pit_pts[inlier_mask]) ** 2, axis=1
            ))
            reproj_err = float(np.mean(errors))
        else:
            reproj_err = float('inf')
    else:
        errors = np.sqrt(np.sum((projected - pit_pts) ** 2, axis=1))
        reproj_err = float(np.mean(errors))

    # Sanity check: project player feet and verify reasonable positions.
    # A correct homography should place most detected persons on or very
    # near the pitch.  Coaches/subs/ball boys may be slightly off, but the
    # majority should land within a small margin of the 105×68 rectangle.
    if player_boxes and len(player_boxes) >= 5:
        feet = []
        for bx1, by1, bx2, by2 in player_boxes:
            feet.append([(bx1 + bx2) / 2, by2])
        feet_arr = np.array(feet, dtype=np.float64).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(feet_arr, H).reshape(-1, 2)
        # Count players that project within 5m of the pitch boundary
        on_pitch = 0
        for mx, my in mapped:
            if -5 <= mx <= PITCH_LENGTH + 5 and -5 <= my <= PITCH_WIDTH + 5:
                on_pitch += 1
        frac_on = on_pitch / len(player_boxes)
        # If fewer than 35% of players project near the pitch, reject
        if frac_on < 0.35:
            return HomographyResult(H=None, lines=lines,
                                    n_matches=len(image_points))

    return HomographyResult(
        H=H,
        n_inliers=n_inliers,
        n_matches=len(image_points),
        reprojection_error=reproj_err,
        image_points=img_pts,
        pitch_points=pit_pts,
        point_labels=point_labels,
        lines=lines,
    )


# ── Visualization helpers ────────────────────────────────────────────

def draw_debug(
    frame: np.ndarray,
    field_mask: np.ndarray,
    line_mask: np.ndarray,
    lines: List[DetectedLine],
    near_tl: Optional[DetectedLine] = None,
    far_tl: Optional[DetectedLine] = None,
) -> np.ndarray:
    """Draw a 2x2 debug visualization."""
    h, w = frame.shape[:2]

    # Panel 1: original
    p1 = frame.copy()

    # Panel 2: field mask
    p2 = frame.copy()
    overlay = np.zeros_like(frame)
    overlay[:, :, 1] = field_mask
    p2 = cv2.addWeighted(p2, 0.7, overlay, 0.3, 0)

    # Panel 3: line mask
    p3 = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)

    # Panel 4: classified lines
    p4 = frame.copy()
    for l in lines:
        if l.label == "touchline":
            color = (0, 255, 255)  # yellow
        elif l.label == "perpendicular":
            color = (255, 0, 255)  # magenta
        elif l.label == "other":
            color = (255, 165, 0)  # orange
        else:
            color = (0, 255, 0)    # green
        thickness = 3 if l in (near_tl, far_tl) else 2
        cv2.line(p4, (l.x1, l.y1), (l.x2, l.y2), color, thickness)

    # Draw touchline intersections with non-touchline lines
    perps = [l for l in lines if l.label in ("perpendicular", "other")]
    for tl in [near_tl, far_tl]:
        if tl is None:
            continue
        for pl in perps:
            pt = _line_intersection(pl, tl)
            if pt is None:
                continue
            x, y = pt
            if -50 <= x <= w + 50 and -50 <= y <= h + 50:
                cv2.circle(p4, (int(x), int(y)), 8, (0, 0, 255), -1)
                cv2.circle(p4, (int(x), int(y)), 10, (255, 255, 255), 2)

    n_tl = sum(1 for l in lines if l.label == "touchline")
    n_perp = sum(1 for l in lines if l.label == "perpendicular")
    n_other = sum(1 for l in lines if l.label == "other")
    cv2.putText(p4, f"TL:{n_tl} Perp:{n_perp} Other:{n_other}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if near_tl:
        cv2.putText(p4, "NEAR", (near_tl.x1, near_tl.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if far_tl:
        cv2.putText(p4, "FAR", (far_tl.x1, far_tl.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    return np.vstack([top, bot])
