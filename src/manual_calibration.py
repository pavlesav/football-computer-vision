"""
Manual pitch keypoint annotation + homography reference frames.

For matches where PnLCalib consistently fails (Tier 3: night/dusk games,
oblique cameras, running tracks), the user can annotate 4+ pitch
landmarks on a small number of reference frames. The resulting plane
homographies are then used by the video runner to:

  1. Seed the optical-flow tracker when PnLCalib hasn't produced a
     trusted projection yet.
  2. Provide a last-resort fallback during long inference gaps.

Saved files live under ``data/manual_calibration/{slug}_frame_{N:07d}.json``
and are picked up automatically by ``load_match_seeds(slug)``.

The notebook widget (see ``02_homography.ipynb``) drives this module
for interactive annotation. ``compute_homography`` / ``save_calibration``
are pure functions callable from any context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Named planar pitch lines, each as ((wx1, wy1), (wx2, wy2)) in natural pitch
# coords (0..105 x 0..68). Drives the line-drag labeling widget — picking a name
# from the dropdown highlights the line on the pitch diagram, and the segment
# the user drags on the image is paired with this world line.
NAMED_PITCH_LINES: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "Halfway":         ((PITCH_LENGTH / 2, 0.0), (PITCH_LENGTH / 2, PITCH_WIDTH)),
    "Top touchline":   ((0.0, 0.0), (PITCH_LENGTH, 0.0)),
    "Bot touchline":   ((0.0, PITCH_WIDTH), (PITCH_LENGTH, PITCH_WIDTH)),
    "L goal line":     ((0.0, 0.0), (0.0, PITCH_WIDTH)),
    "R goal line":     ((PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH)),
    "L PA front":      ((16.5, 13.84), (16.5, 54.16)),
    "L PA top":        ((0.0, 13.84), (16.5, 13.84)),
    "L PA bot":        ((0.0, 54.16), (16.5, 54.16)),
    "R PA front":      ((PITCH_LENGTH - 16.5, 13.84), (PITCH_LENGTH - 16.5, 54.16)),
    "R PA top":        ((PITCH_LENGTH - 16.5, 13.84), (PITCH_LENGTH, 13.84)),
    "R PA bot":        ((PITCH_LENGTH - 16.5, 54.16), (PITCH_LENGTH, 54.16)),
    "L 6yd front":     ((5.5, 24.84), (5.5, 43.16)),
    "L 6yd top":       ((0.0, 24.84), (5.5, 24.84)),
    "L 6yd bot":       ((0.0, 43.16), (5.5, 43.16)),
    "R 6yd front":     ((PITCH_LENGTH - 5.5, 24.84), (PITCH_LENGTH - 5.5, 43.16)),
    "R 6yd top":       ((PITCH_LENGTH - 5.5, 24.84), (PITCH_LENGTH, 24.84)),
    "R 6yd bot":       ((PITCH_LENGTH - 5.5, 43.16), (PITCH_LENGTH, 43.16)),
}

# Named single-point landmarks that sit on the painted curves where they cross
# a straight line — useful in close-up frames where 4 distinct straight lines
# aren't all visible. The widget lets the user TAP these (no drag) and adds them
# as point correspondences alongside line ones.
#
# Center-circle ∩ halfway line (x = 52.5):
#   (52.5, 34 ± 9.15)
# Penalty-arc ∩ PA-front (left arc has center (11, 34), r=9.15 → at x=16.5,
# y = 34 ± sqrt(9.15² - 5.5²) = 34 ± 7.3127):
_ARC_DY = float(np.sqrt(9.15 ** 2 - 5.5 ** 2))
ARC_INTERSECTION_POINTS: Dict[str, Tuple[float, float]] = {
    "Halfway ∩ CC top":   (PITCH_LENGTH / 2, PITCH_WIDTH / 2 - 9.15),
    "Halfway ∩ CC bot":   (PITCH_LENGTH / 2, PITCH_WIDTH / 2 + 9.15),
    "L PA ∩ arc top":     (16.5, PITCH_WIDTH / 2 - _ARC_DY),
    "L PA ∩ arc bot":     (16.5, PITCH_WIDTH / 2 + _ARC_DY),
    "R PA ∩ arc top":     (PITCH_LENGTH - 16.5, PITCH_WIDTH / 2 - _ARC_DY),
    "R PA ∩ arc bot":     (PITCH_LENGTH - 16.5, PITCH_WIDTH / 2 + _ARC_DY),
    "Center spot":        (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    "L penalty spot":     (11.0, PITCH_WIDTH / 2),
    "R penalty spot":     (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),
}

# Canonical pitch landmarks — name -> (x_m, y_m) in natural 0..105 x 0..68.
# The labeling widget dropdown is driven by this list. Keep names short and
# unambiguous; users need to identify them at a glance on broadcast footage.
KEY_PITCH_POINTS: Dict[str, Tuple[float, float]] = {
    # Corners
    "TL_corner":        (0.0, 0.0),
    "TR_corner":        (PITCH_LENGTH, 0.0),
    "BL_corner":        (0.0, PITCH_WIDTH),
    "BR_corner":        (PITCH_LENGTH, PITCH_WIDTH),
    # Halfway line
    "Halfway_top":      (PITCH_LENGTH / 2, 0.0),
    "Halfway_bottom":   (PITCH_LENGTH / 2, PITCH_WIDTH),
    "Center":           (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    # Center circle <-> halfway line
    "Center_circle_top":    (PITCH_LENGTH / 2, PITCH_WIDTH / 2 - 9.15),
    "Center_circle_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH / 2 + 9.15),
    # Left penalty area (x=16.5)
    "L_PA_outer_top":    (16.5, 13.84),
    "L_PA_outer_bottom": (16.5, 54.16),
    "L_PA_goal_top":     (0.0, 13.84),
    "L_PA_goal_bottom":  (0.0, 54.16),
    # Left goal area (x=5.5)
    "L_GA_outer_top":    (5.5, 24.84),
    "L_GA_outer_bottom": (5.5, 43.16),
    "L_GA_goal_top":     (0.0, 24.84),
    "L_GA_goal_bottom":  (0.0, 43.16),
    # Right penalty area (x=88.5)
    "R_PA_outer_top":    (88.5, 13.84),
    "R_PA_outer_bottom": (88.5, 54.16),
    "R_PA_goal_top":     (PITCH_LENGTH, 13.84),
    "R_PA_goal_bottom":  (PITCH_LENGTH, 54.16),
    # Right goal area (x=99.5)
    "R_GA_outer_top":    (99.5, 24.84),
    "R_GA_outer_bottom": (99.5, 43.16),
    "R_GA_goal_top":     (PITCH_LENGTH, 24.84),
    "R_GA_goal_bottom":  (PITCH_LENGTH, 43.16),
    # Penalty spots
    "L_penalty_spot":    (11.0, PITCH_WIDTH / 2),
    "R_penalty_spot":    (94.0, PITCH_WIDTH / 2),
}


# ── IO ─────────────────────────────────────────────────────────────────

def calibration_dir() -> Path:
    d = Config.PROJECT_ROOT / "data" / "manual_calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def calibration_path(slug: str, frame_number: int) -> Path:
    return calibration_dir() / f"{slug}_frame_{frame_number:07d}.json"


# ── Homography helpers ─────────────────────────────────────────────────

def compute_homography(
    correspondences: List[Tuple[str, Tuple[float, float], Tuple[float, float]]],
) -> Optional[np.ndarray]:
    """Compute image->pitch (natural 0..105 x 0..68) plane homography.

    Parameters
    ----------
    correspondences : list of (landmark_name, pitch_xy_m, image_xy_px)

    Returns
    -------
    3x3 homography H such that ``H @ [px, py, 1]`` yields homogeneous
    ``(wx, wy, 1)`` in pitch metres. Returns ``None`` if fewer than 4
    points are given or if the solver fails.
    """
    if len(correspondences) < 4:
        return None
    image_pts = np.array([c[2] for c in correspondences], dtype=np.float64)
    pitch_pts = np.array([c[1] for c in correspondences], dtype=np.float64)
    # With only 4 exact points we want LS, not RANSAC; with more, RANSAC
    # rejects mislabeled clicks.
    method = cv2.RANSAC if len(correspondences) >= 6 else 0
    H, _ = cv2.findHomography(image_pts, pitch_pts, method=method, ransacReprojThreshold=3.0)
    return H


def reprojection_error(
    H: np.ndarray,
    correspondences: List[Tuple[str, Tuple[float, float], Tuple[float, float]]],
) -> float:
    """Mean euclidean reprojection error in metres (image -> pitch)."""
    if H is None:
        return float("inf")
    errs = []
    for _name, (wx, wy), (px, py) in correspondences:
        h = H @ np.array([px, py, 1.0])
        if abs(h[2]) < 1e-10:
            continue
        errs.append(np.hypot(h[0] / h[2] - wx, h[1] / h[2] - wy))
    return float(np.mean(errs)) if errs else float("inf")


def homography_from_lines_and_points(
    image_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    world_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    image_points: Optional[List[Tuple[float, float]]] = None,
    world_points: Optional[List[Tuple[float, float]]] = None,
) -> Optional[np.ndarray]:
    """Image -> pitch homography from a mix of line and point correspondences.

    Both kinds give 2 linear constraints on M = H^(-T):

      * Line: ``cross(l_world, M @ l_image) = 0`` (l in homogeneous form).
      * Point: ``cross(p_image, M^T @ p_world) = 0`` (rearranged from
        ``p_world ∝ M^(-T) @ p_image``).

    SVD nullspace of the stacked constraint matrix gives M; H = M^(-T).
    Each correspondence (line or point) contributes 2 rows; we need 4 total
    correspondences (8 rows) to fit M up to scale.
    """
    image_points = image_points or []
    world_points = world_points or []
    if len(image_segments) != len(world_segments):
        return None
    if len(image_points) != len(world_points):
        return None

    # Minimum-correspondence rule:
    #   Point-only:  >= 4 points   (rank-8 system in 9D, unique nullspace)
    #   Any lines:   >= 5 total    (4 lines alone leave a 2-D nullspace because
    #                               singular rank-1 Ms also satisfy the line
    #                               constraint set; one extra correspondence —
    #                               another line, or a point — breaks it).
    n_lines = len(image_segments)
    n_points = len(image_points)
    if n_lines == 0:
        if n_points < 4:
            return None
    else:
        if n_lines + n_points < 5:
            return None

    rows: List[List[float]] = []

    for (p1, p2), (w1, w2) in zip(image_segments, world_segments):
        li = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
        lw = np.cross([w1[0], w1[1], 1.0], [w2[0], w2[1], 1.0])
        ni, nw = float(np.linalg.norm(li)), float(np.linalg.norm(lw))
        if ni < 1e-10 or nw < 1e-10:
            continue
        li = li / ni
        lw = lw / nw
        a, b, c = lw
        x, y, z = li
        rows.append([0.0, 0.0, 0.0, -c * x, -c * y, -c * z, b * x, b * y, b * z])
        rows.append([c * x, c * y, c * z, 0.0, 0.0, 0.0, -a * x, -a * y, -a * z])

    for (px, py), (wx, wy) in zip(image_points, world_points):
        # cross(p_image, M^T p_world) = 0 — using two of the three independent
        # rows. With p_image = [px, py, 1] and p_world_h = [wx, wy, 1]:
        rows.append([0.0,        -wx,    py * wx,
                     0.0,        -wy,    py * wy,
                     0.0,        -1.0,   py])
        rows.append([wx,         0.0,   -px * wx,
                     wy,         0.0,   -px * wy,
                     1.0,        0.0,   -px])

    if len(rows) < 8:
        return None
    A = np.asarray(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 3)
    try:
        H = np.linalg.inv(M.T)
    except np.linalg.LinAlgError:
        return None
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    return H


def homography_from_line_pairs(
    image_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    world_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Optional[np.ndarray]:
    """Image -> pitch homography from line-only correspondences.

    Thin wrapper for ``homography_from_lines_and_points`` with no point input —
    kept for backward compatibility with the line-only widget.
    """
    if len(image_segments) < 4:
        return None
    return homography_from_lines_and_points(image_segments, world_segments)


def line_alignment_error_px(
    image_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    world_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    H: np.ndarray,
) -> float:
    """Mean perpendicular pixel distance from each projected world-line endpoint
    to the user-drawn image line (extended infinitely).

    Smaller is better. Used for live "this fit is good / sketchy / wrong"
    feedback in the labeling widget.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf")

    errors: List[float] = []
    for (p1, p2), (w1, w2) in zip(image_segments, world_segments):
        l_i = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
        denom = float(np.hypot(l_i[0], l_i[1]))
        if denom < 1e-10:
            continue
        l_i = l_i / denom
        for w in (w1, w2):
            wh = H_inv @ np.array([w[0], w[1], 1.0])
            if abs(wh[2]) < 1e-10:
                continue
            ix, iy = wh[0] / wh[2], wh[1] / wh[2]
            errors.append(abs(l_i[0] * ix + l_i[1] * iy + l_i[2]))
    return float(np.mean(errors)) if errors else float("inf")


def projection_from_natural_homography(H_img_to_pitch_natural: np.ndarray) -> np.ndarray:
    """Convert a natural-coords image->pitch 3x3 into a PnLCalib-convention 3x4 P.

    PnLCalib's P operates on origin-centered world points (subtract 52.5, 34).
    We therefore:
      1. Shift the homography so its output is origin-centered.
      2. Invert to get pitch->image.
      3. Pack as 3x4 with a zero z-column (planar — goal posts won't render).
    """
    shift = np.array([[1.0, 0.0, -PITCH_LENGTH / 2],
                      [0.0, 1.0, -PITCH_WIDTH / 2],
                      [0.0, 0.0, 1.0]])
    H_img_to_pitch_centered = shift @ H_img_to_pitch_natural
    H_pitch_to_img = np.linalg.inv(H_img_to_pitch_centered)
    P = np.zeros((3, 4), dtype=np.float64)
    P[:, 0] = H_pitch_to_img[:, 0]
    P[:, 1] = H_pitch_to_img[:, 1]
    P[:, 3] = H_pitch_to_img[:, 2]
    return P


# ── Persistence ────────────────────────────────────────────────────────

@dataclass
class ManualCalibration:
    slug: str
    frame_number: int
    frame_width: int
    frame_height: int
    # Point correspondences (legacy click-landmark widget). Empty for
    # line-based calibrations.
    correspondences: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list
    )
    H_image_to_pitch_natural: Optional[List[List[float]]] = None
    P_pnlcalib_convention: Optional[List[List[float]]] = None
    reprojection_error_m: Optional[float] = None
    # Line correspondences (drag-line widget). Each entry:
    #   {"line_name": str,
    #    "image_endpoints": [[x1,y1],[x2,y2]],
    #    "world_endpoints": [[wx1,wy1],[wx2,wy2]]}
    line_correspondences: List[Dict[str, Any]] = field(default_factory=list)
    line_alignment_error_px: Optional[float] = None


def save_calibration(
    slug: str,
    frame_number: int,
    frame_width: int,
    frame_height: int,
    correspondences: List[Tuple[str, Tuple[float, float], Tuple[float, float]]],
) -> Path:
    H = compute_homography(correspondences)
    err = reprojection_error(H, correspondences) if H is not None else None
    P = projection_from_natural_homography(H) if H is not None else None
    data = ManualCalibration(
        slug=slug,
        frame_number=frame_number,
        frame_width=frame_width,
        frame_height=frame_height,
        correspondences=[(name, list(pit), list(img)) for name, pit, img in correspondences],
        H_image_to_pitch_natural=H.tolist() if H is not None else None,
        P_pnlcalib_convention=P.tolist() if P is not None else None,
        reprojection_error_m=err,
    )
    path = calibration_path(slug, frame_number)
    path.write_text(json.dumps(asdict(data), indent=2))
    return path


def load_calibration(path: Path) -> ManualCalibration:
    raw = json.loads(Path(path).read_text())
    valid = {f.name for f in fields(ManualCalibration)}
    return ManualCalibration(**{k: v for k, v in raw.items() if k in valid})


def save_line_calibration(
    slug: str,
    frame_number: int,
    frame_width: int,
    frame_height: int,
    line_correspondences: List[
        Tuple[str, Tuple[Tuple[float, float], Tuple[float, float]],
              Tuple[Tuple[float, float], Tuple[float, float]]]
    ],
    point_correspondences: Optional[
        List[Tuple[str, Tuple[float, float], Tuple[float, float]]]
    ] = None,
) -> Path:
    """Persist a line+point calibration alongside the existing format.

    Computes H + P from the combined line and point pairs and writes the same
    on-disk schema as ``save_calibration`` so ``load_match_seeds`` picks the
    file up unchanged. Point correspondences are stored in the legacy
    ``correspondences`` field; lines go in ``line_correspondences``.
    """
    point_correspondences = point_correspondences or []
    image_segs = [c[1] for c in line_correspondences]
    world_segs = [c[2] for c in line_correspondences]
    image_pts = [c[2] for c in point_correspondences]   # (image_x, image_y)
    world_pts = [c[1] for c in point_correspondences]   # (world_x, world_y)
    H = homography_from_lines_and_points(image_segs, world_segs, image_pts, world_pts)
    err = line_alignment_error_px(image_segs, world_segs, H) if H is not None else None
    P = projection_from_natural_homography(H) if H is not None else None
    data = ManualCalibration(
        slug=slug,
        frame_number=frame_number,
        frame_width=frame_width,
        frame_height=frame_height,
        correspondences=[(name, list(pit), list(img))
                          for name, pit, img in point_correspondences],
        H_image_to_pitch_natural=H.tolist() if H is not None else None,
        P_pnlcalib_convention=P.tolist() if P is not None else None,
        reprojection_error_m=None,
        line_correspondences=[
            {
                "line_name": name,
                "image_endpoints": [list(p1), list(p2)],
                "world_endpoints": [list(w1), list(w2)],
            }
            for name, (p1, p2), (w1, w2) in line_correspondences
        ],
        line_alignment_error_px=err,
    )
    path = calibration_path(slug, frame_number)
    path.write_text(json.dumps(asdict(data), indent=2))
    return path


def load_match_seeds(slug: str) -> Dict[int, np.ndarray]:
    """Return ``{frame_number: P (3x4)}`` for all saved frames of this match."""
    result: Dict[int, np.ndarray] = {}
    for path in calibration_dir().glob(f"{slug}_frame_*.json"):
        try:
            cal = load_calibration(path)
        except Exception:
            continue
        if cal.P_pnlcalib_convention is None:
            continue
        result[int(cal.frame_number)] = np.array(cal.P_pnlcalib_convention, dtype=np.float64)
    return result


# ── Pitch diagram (for widget overlay) ─────────────────────────────────

def draw_pitch_diagram(ax, point_highlight: Optional[str] = None) -> None:
    """Render a 105x68 m pitch schematic on the given matplotlib axes.

    If ``point_highlight`` is a key of ``KEY_PITCH_POINTS``, that landmark
    is drawn as a large red dot so the user can see what to click.
    """
    ax.set_xlim(-3, PITCH_LENGTH + 3)
    ax.set_ylim(PITCH_WIDTH + 3, -3)   # invert Y: (0,0) top-left, matches image convention
    ax.set_aspect("equal")
    ax.set_facecolor("#2e7d32")

    lw = 1.2
    green = "white"

    # Outer rectangle
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0],
            [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color=green, lw=lw)
    # Halfway line
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color=green, lw=lw)
    # Center circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(PITCH_LENGTH / 2 + 9.15 * np.cos(theta),
            PITCH_WIDTH / 2 + 9.15 * np.sin(theta), color=green, lw=lw)
    ax.plot(PITCH_LENGTH / 2, PITCH_WIDTH / 2, "o", color=green, markersize=3)
    # Penalty areas
    for x_out, x_in in [(0, 16.5), (PITCH_LENGTH, PITCH_LENGTH - 16.5)]:
        ax.plot([x_out, x_in, x_in, x_out],
                [13.84, 13.84, 54.16, 54.16], color=green, lw=lw)
    # Goal areas
    for x_out, x_in in [(0, 5.5), (PITCH_LENGTH, PITCH_LENGTH - 5.5)]:
        ax.plot([x_out, x_in, x_in, x_out],
                [24.84, 24.84, 43.16, 43.16], color=green, lw=lw)
    # Penalty spots
    for x in (11.0, PITCH_LENGTH - 11.0):
        ax.plot(x, PITCH_WIDTH / 2, "o", color=green, markersize=3)

    if point_highlight and point_highlight in KEY_PITCH_POINTS:
        hx, hy = KEY_PITCH_POINTS[point_highlight]
        ax.plot(hx, hy, "o", color="red", markersize=14, markeredgecolor="white",
                markeredgewidth=2)
        ax.annotate(point_highlight, (hx, hy), color="red", fontsize=9,
                    xytext=(6, -6), textcoords="offset points", fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])


def draw_pitch_diagram_with_line(
    ax,
    line_highlight: Optional[str] = None,
    point_highlight: Optional[str] = None,
    drawn_lines: Optional[List[str]] = None,
    drawn_points: Optional[List[str]] = None,
) -> None:
    """Pitch schematic with selected/drawn lines and points marked.

    - ``line_highlight``: name from NAMED_PITCH_LINES drawn in red (the line
      the user is about to drag).
    - ``point_highlight``: name from ARC_INTERSECTION_POINTS drawn as a large
      red dot (the point the user is about to tap).
    - ``drawn_lines`` / ``drawn_points``: already-recorded names; shown as
      lime-green markers so the user sees what's already covered.
    """
    draw_pitch_diagram(ax)
    if drawn_lines:
        for name in drawn_lines:
            if name not in NAMED_PITCH_LINES:
                continue
            (wx1, wy1), (wx2, wy2) = NAMED_PITCH_LINES[name]
            ax.plot([(wx1 + wx2) / 2], [(wy1 + wy2) / 2], "s",
                    color="lime", markersize=8, markeredgecolor="black")
    if drawn_points:
        for name in drawn_points:
            if name not in ARC_INTERSECTION_POINTS:
                continue
            wx, wy = ARC_INTERSECTION_POINTS[name]
            ax.plot([wx], [wy], "o", color="lime", markersize=8,
                    markeredgecolor="black")
    if line_highlight and line_highlight in NAMED_PITCH_LINES:
        (wx1, wy1), (wx2, wy2) = NAMED_PITCH_LINES[line_highlight]
        ax.plot([wx1, wx2], [wy1, wy2], color="red", lw=4, alpha=0.85)
        mx, my = (wx1 + wx2) / 2, (wy1 + wy2) / 2
        ax.annotate(line_highlight, (mx, my), color="red", fontsize=10,
                    xytext=(8, -8), textcoords="offset points", fontweight="bold")
    if point_highlight and point_highlight in ARC_INTERSECTION_POINTS:
        wx, wy = ARC_INTERSECTION_POINTS[point_highlight]
        ax.plot(wx, wy, "o", color="red", markersize=14,
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(point_highlight, (wx, wy), color="red", fontsize=9,
                    xytext=(6, -6), textcoords="offset points", fontweight="bold")


# ── Interactive widget (optional) ──────────────────────────────────────

def build_labeling_widget(
    slug: str,
    frame: np.ndarray,
    frame_number: int,
    existing: Optional[ManualCalibration] = None,
):
    """Create an ipywidgets UI for labeling pitch keypoints on a frame.

    Layout:
      - Left: video frame (click to record the image point of the
              currently selected landmark).
      - Right: pitch diagram with the currently selected landmark
              highlighted.
      - Below: dropdown to pick the next landmark, buttons for
              Remove / Clear / Save, live reprojection-error readout.

    Requires ``%matplotlib widget`` (ipympl). Returns a VBox widget
    that you display() from the notebook cell.
    """
    # Imports local so this module stays importable headless.
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    H, W = frame.shape[:2]

    correspondences: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = []
    if existing is not None:
        for name, pit, img in existing.correspondences:
            correspondences.append((name, tuple(pit), tuple(img)))

    landmark_options = list(KEY_PITCH_POINTS.keys())
    dropdown = widgets.Dropdown(
        options=landmark_options,
        value=landmark_options[0],
        description="Landmark:",
        layout=widgets.Layout(width="340px"),
    )
    status = widgets.HTML(value="")
    btn_remove = widgets.Button(description="Remove last", icon="undo")
    btn_clear = widgets.Button(description="Clear all", icon="trash",
                               button_style="warning")
    btn_save = widgets.Button(description="Save", icon="save",
                              button_style="success")

    fig, (ax_img, ax_pitch) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [W / H, 2.0]},
    )
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.footer_visible = False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax_img.imshow(rgb)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"{slug}  frame {frame_number}")

    def redraw_pitch():
        ax_pitch.clear()
        draw_pitch_diagram(ax_pitch, dropdown.value)
        # Overlay already-clicked landmarks
        for name, (wx, wy), _img in correspondences:
            ax_pitch.plot(wx, wy, "o", color="yellow", markersize=8,
                          markeredgecolor="black")
        fig.canvas.draw_idle()

    def redraw_image_overlay():
        # Remove all artist children except the imshow (first one)
        for line in list(ax_img.lines):
            line.remove()
        for txt in list(ax_img.texts):
            txt.remove()
        for name, _pit, (px, py) in correspondences:
            ax_img.plot(px, py, "o", color="yellow", markersize=8,
                        markeredgecolor="black")
            ax_img.annotate(name, (px, py), color="yellow", fontsize=7,
                            xytext=(5, -5), textcoords="offset points")
        fig.canvas.draw_idle()

    def update_status():
        n = len(correspondences)
        if n < 4:
            status.value = (f"<b>{n}/4 points</b> — need at least 4 to compute "
                            "a homography.")
            return
        H_mat = compute_homography(correspondences)
        err = reprojection_error(H_mat, correspondences)
        color = "green" if err < 1.5 else ("orange" if err < 3.0 else "red")
        status.value = (f"<b>{n} points</b> — mean reprojection error: "
                        f"<span style='color:{color}'>{err:.2f} m</span>")

    def on_click(event):
        if event.inaxes is not ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        name = dropdown.value
        # Replace if this landmark was already clicked
        correspondences[:] = [c for c in correspondences if c[0] != name]
        correspondences.append((name, KEY_PITCH_POINTS[name],
                                (float(event.xdata), float(event.ydata))))
        # Advance to next landmark for convenience
        idx = landmark_options.index(name)
        next_idx = (idx + 1) % len(landmark_options)
        dropdown.value = landmark_options[next_idx]
        redraw_image_overlay()
        redraw_pitch()
        update_status()

    def on_dropdown_change(change):
        if change["name"] == "value":
            redraw_pitch()

    def on_remove(_b):
        if correspondences:
            correspondences.pop()
            redraw_image_overlay()
            redraw_pitch()
            update_status()

    def on_clear(_b):
        correspondences.clear()
        redraw_image_overlay()
        redraw_pitch()
        update_status()

    def on_save(_b):
        if len(correspondences) < 4:
            status.value = "<b style='color:red'>Need 4+ points before saving.</b>"
            return
        path = save_calibration(slug, frame_number, W, H, correspondences)
        status.value = (f"Saved to <code>{path.relative_to(Config.PROJECT_ROOT)}</code>. "
                        f"{status.value}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    dropdown.observe(on_dropdown_change, names="value")
    btn_remove.on_click(on_remove)
    btn_clear.on_click(on_clear)
    btn_save.on_click(on_save)

    redraw_pitch()
    redraw_image_overlay()
    update_status()

    controls = widgets.HBox([dropdown, btn_remove, btn_clear, btn_save])
    return widgets.VBox([controls, status])


def build_line_labeling_widget(
    slug: str,
    frame: np.ndarray,
    frame_number: int,
    existing: Optional[ManualCalibration] = None,
    tap_threshold_px: float = 10.0,
):
    """Drag-line + tap-point UI for ground-truth pitch homography.

    The widget accepts two kinds of correspondences in the same canvas:

    - **Drag** a line: pick a line name from the *Line:* dropdown, click on
      one end of the painted line in the frame and drag past the other end.
      Endpoints don't need to be precise — only the line through them
      contributes. Useful for halfway, touchlines, PA edges, 6yd edges.
    - **Tap** a point: pick a name from the *Point:* dropdown and single-click
      on the painted intersection (no drag). Useful where straight lines are
      scarce — e.g. center-circle ∩ halfway, penalty-arc ∩ PA-front.

    Tap-vs-drag is auto-detected by movement distance: anything < ~10 px is
    treated as a tap.

    Once at least 5 correspondences are recorded (any mix of lines and
    points), every named pitch line is back-projected onto the frame in
    **cyan** as a live fit preview. **Save** writes the same JSON schema as
    the click-landmark widget — line correspondences in
    ``line_correspondences``, point correspondences in ``correspondences``.

    Requires ``%matplotlib widget`` (ipympl).
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    H_img, W_img = frame.shape[:2]
    line_names = list(NAMED_PITCH_LINES.keys())
    point_names = list(ARC_INTERSECTION_POINTS.keys())

    # Two parallel lists of recorded correspondences.
    #   lines:  (line_name, ((px1,py1),(px2,py2)), ((wx1,wy1),(wx2,wy2)))
    #   points: (point_name, (wx,wy), (px,py))   — pitch_xy then image_xy
    line_corrs: List[
        Tuple[str, Tuple[Tuple[float, float], Tuple[float, float]],
              Tuple[Tuple[float, float], Tuple[float, float]]]
    ] = []
    point_corrs: List[Tuple[str, Tuple[float, float], Tuple[float, float]]] = []

    if existing is not None:
        for entry in existing.line_correspondences or []:
            (p1, p2) = entry["image_endpoints"]
            (w1, w2) = entry["world_endpoints"]
            line_corrs.append((entry["line_name"],
                                ((float(p1[0]), float(p1[1])),
                                 (float(p2[0]), float(p2[1]))),
                                ((float(w1[0]), float(w1[1])),
                                 (float(w2[0]), float(w2[1])))))
        for name, pit, img in existing.correspondences or []:
            if name in ARC_INTERSECTION_POINTS:
                point_corrs.append((name,
                                    (float(pit[0]), float(pit[1])),
                                    (float(img[0]), float(img[1]))))

    line_dd = widgets.Dropdown(
        options=line_names, value=line_names[0],
        description="Line:", layout=widgets.Layout(width="260px"),
    )
    point_dd = widgets.Dropdown(
        options=point_names, value=point_names[0],
        description="Point:", layout=widgets.Layout(width="260px"),
    )
    status = widgets.HTML()
    btn_remove = widgets.Button(description="Remove last", icon="undo",
                                layout=widgets.Layout(width="auto"))
    btn_clear = widgets.Button(description="Clear all", icon="trash",
                                button_style="warning",
                                layout=widgets.Layout(width="auto"))
    btn_save = widgets.Button(description="Save", icon="save",
                                button_style="success",
                                layout=widgets.Layout(width="auto"))

    fig, (ax_img, ax_pitch) = plt.subplots(
        1, 2, figsize=(15, 5),
        gridspec_kw={"width_ratios": [W_img / H_img, 1.4]},
    )
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.footer_visible = False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax_img.imshow(rgb)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"{slug}  frame {frame_number}  —  drag a line, or tap a "
                      "named intersection point")

    # Reusable Line2D for cheap drag preview during motion.
    (preview_line,) = ax_img.plot([], [], "-", color="yellow", lw=2.0, alpha=0.8)

    # Order in which correspondences were added — drives "remove last" undo.
    add_order: List[str] = ["line"] * len(line_corrs) + ["point"] * len(point_corrs)
    drag_state = {"start": None, "active": False}

    LINE_COLORS = plt.cm.tab10.colors

    def _next_unused(values: List[str], known: List[str]) -> Optional[str]:
        used = set(values)
        for nm in known:
            if nm not in used:
                return nm
        return None

    def _redraw_image_overlay():
        # Wipe everything except the imshow and the preview Line2D.
        for ln in list(ax_img.lines):
            if ln is preview_line:
                continue
            ln.remove()
        for tx in list(ax_img.texts):
            tx.remove()

        # Recorded line segments.
        for i, (name, (p1, p2), _w) in enumerate(line_corrs):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            ax_img.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
                        color=color, lw=2.5, alpha=0.95)
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax_img.annotate(name, (mx, my), color=color, fontsize=8,
                            xytext=(5, -5), textcoords="offset points",
                            fontweight="bold")

        # Recorded points.
        for i, (name, _w, (px, py)) in enumerate(point_corrs):
            color = LINE_COLORS[(len(line_corrs) + i) % len(LINE_COLORS)]
            ax_img.plot(px, py, "o", color=color, markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5)
            ax_img.annotate(name, (px, py), color=color, fontsize=8,
                            xytext=(8, -8), textcoords="offset points",
                            fontweight="bold")

        # Live homography preview once we have enough correspondences.
        H_fit = _current_h()
        if H_fit is not None:
            try:
                H_inv = np.linalg.inv(H_fit)
            except np.linalg.LinAlgError:
                H_inv = None
            if H_inv is not None:
                for nm, (w1, w2) in NAMED_PITCH_LINES.items():
                    endpoints = []
                    for w in (w1, w2):
                        wh = H_inv @ np.array([w[0], w[1], 1.0])
                        if abs(wh[2]) < 1e-9:
                            endpoints = []
                            break
                        endpoints.append((wh[0] / wh[2], wh[1] / wh[2]))
                    if len(endpoints) == 2:
                        ax_img.plot(
                            [endpoints[0][0], endpoints[1][0]],
                            [endpoints[0][1], endpoints[1][1]],
                            "-", color="cyan", lw=1.2, alpha=0.6,
                        )
        fig.canvas.draw_idle()

    def _redraw_pitch():
        ax_pitch.clear()
        draw_pitch_diagram_with_line(
            ax_pitch,
            line_highlight=line_dd.value,
            point_highlight=point_dd.value,
            drawn_lines=[c[0] for c in line_corrs],
            drawn_points=[c[0] for c in point_corrs],
        )
        fig.canvas.draw_idle()

    def _current_h() -> Optional[np.ndarray]:
        return homography_from_lines_and_points(
            [c[1] for c in line_corrs],
            [c[2] for c in line_corrs],
            [c[2] for c in point_corrs],
            [c[1] for c in point_corrs],
        )

    def _update_status():
        n_l, n_p = len(line_corrs), len(point_corrs)
        n_total = n_l + n_p
        # Required minimum depends on whether any lines are used.
        need = 4 if n_l == 0 else 5
        if n_total < need:
            hint = ("Tip: drag a couple of pitch lines plus tap a circle/arc "
                    "intersection.")
            status.value = (
                f"<b>{n_total}/{need}</b> &nbsp;(lines: {n_l}, points: {n_p}) "
                f"— {hint}"
            )
            return
        H_fit = _current_h()
        if H_fit is None:
            status.value = (f"<b>{n_total}</b> correspondences — degenerate fit. "
                            "Add a line/point in a different direction.")
            return
        err = line_alignment_error_px(
            [c[1] for c in line_corrs], [c[2] for c in line_corrs], H_fit,
        ) if line_corrs else 0.0
        color = "limegreen" if err < 5 else ("orange" if err < 15 else "red")
        status.value = (f"<b>{n_total}</b> correspondences "
                        f"(lines: {n_l}, points: {n_p}) — mean line distance: "
                        f"<span style='color:{color}'>{err:.1f} px</span>")

    def _on_press(event):
        if event.inaxes is not ax_img or event.xdata is None:
            return
        drag_state["start"] = (float(event.xdata), float(event.ydata))
        drag_state["active"] = True
        preview_line.set_data([event.xdata, event.xdata],
                              [event.ydata, event.ydata])
        fig.canvas.draw_idle()

    def _on_motion(event):
        if not drag_state["active"] or event.inaxes is not ax_img:
            return
        if event.xdata is None:
            return
        x0, y0 = drag_state["start"]
        preview_line.set_data([x0, event.xdata], [y0, event.ydata])
        fig.canvas.draw_idle()

    def _on_release(event):
        if not drag_state["active"]:
            return
        drag_state["active"] = False
        if event.inaxes is not ax_img or event.xdata is None:
            preview_line.set_data([], [])
            fig.canvas.draw_idle()
            return
        x0, y0 = drag_state["start"]
        x1, y1 = float(event.xdata), float(event.ydata)
        dist = float(np.hypot(x1 - x0, y1 - y0))
        preview_line.set_data([], [])
        if dist < tap_threshold_px:
            # TAP — record selected named point at the click location.
            name = point_dd.value
            point_corrs[:] = [c for c in point_corrs if c[0] != name]
            point_corrs.append((name, ARC_INTERSECTION_POINTS[name],
                                (x0, y0)))
            add_order.append("point")
            nxt = _next_unused([c[0] for c in point_corrs], point_names)
            if nxt is not None:
                point_dd.value = nxt
        else:
            # DRAG — record selected named line through the two clicked points.
            name = line_dd.value
            line_corrs[:] = [c for c in line_corrs if c[0] != name]
            line_corrs.append((name,
                                ((x0, y0), (x1, y1)),
                                NAMED_PITCH_LINES[name]))
            add_order.append("line")
            nxt = _next_unused([c[0] for c in line_corrs], line_names)
            if nxt is not None:
                line_dd.value = nxt
        _redraw_image_overlay()
        _redraw_pitch()
        _update_status()

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)

    def _on_dropdown(_change):
        _redraw_pitch()

    def _on_remove(_b):
        if not add_order:
            return
        kind = add_order.pop()
        if kind == "line" and line_corrs:
            line_corrs.pop()
        elif kind == "point" and point_corrs:
            point_corrs.pop()
        _redraw_image_overlay()
        _redraw_pitch()
        _update_status()

    def _on_clear(_b):
        line_corrs.clear()
        point_corrs.clear()
        add_order.clear()
        _redraw_image_overlay()
        _redraw_pitch()
        _update_status()

    def _on_save(_b):
        n_l, n_p = len(line_corrs), len(point_corrs)
        n_total = n_l + n_p
        need = 4 if n_l == 0 else 5
        if n_total < need:
            status.value = (f"<span style='color:red'>Need at least {need} "
                            f"correspondences (have {n_total}).</span>")
            return
        H_fit = _current_h()
        if H_fit is None:
            status.value = ("<span style='color:red'>Correspondences are "
                            "degenerate — add one in a different direction."
                            "</span>")
            return
        path = save_line_calibration(
            slug, frame_number, W_img, H_img, line_corrs, point_corrs,
        )
        err = line_alignment_error_px(
            [c[1] for c in line_corrs], [c[2] for c in line_corrs], H_fit,
        ) if line_corrs else 0.0
        status.value = (f"<b>Saved</b> {n_total} correspondences to "
                        f"{path.name}  —  alignment error {err:.1f} px.")

    line_dd.observe(_on_dropdown, names="value")
    point_dd.observe(_on_dropdown, names="value")
    btn_remove.on_click(_on_remove)
    btn_clear.on_click(_on_clear)
    btn_save.on_click(_on_save)

    _redraw_image_overlay()
    _redraw_pitch()
    _update_status()

    return widgets.VBox([
        widgets.HBox([line_dd, point_dd]),
        widgets.HBox([btn_remove, btn_clear, btn_save]),
        status,
    ])


def build_line_adjust_widget(
    slug: str,
    frame: np.ndarray,
    frame_number: int,
    initial_P: np.ndarray,
    existing: Optional[ManualCalibration] = None,
    handle_radius_px: float = 14.0,
    line_pick_px: float = 8.0,
):
    """Adjust the pipeline's projected pitch lines into ground truth.

    All NAMED_PITCH_LINES are pre-projected onto the frame using ``initial_P``
    (3x4, PnLCalib convention — same one ``project_lines`` consumes). Lines
    start unconfirmed (faded gray). The user:

    - **Clicks on a line** (~within ``line_pick_px``) to confirm it at the
      pipeline's current projection. Confirmed lines turn solid coloured and
      get two endpoint handles.
    - **Drags an endpoint handle** to nudge that endpoint anywhere — only the
      line through the two endpoints matters, so the handle can be placed
      wherever the painted line is most visible.
    - **Confirm all** marks every line as confirmed at its current position
      (useful when the pipeline H is mostly right and only a few lines need
      correction).
    - **Reset** discards edits and re-projects from ``initial_P``.

    Once 4+ lines are confirmed, every NAMED_PITCH_LINE is re-projected in
    cyan as a live preview of the fit. **Save** writes the same JSON schema
    as ``build_line_labeling_widget`` — load_match_seeds() picks it up
    unchanged.

    Requires ``%matplotlib widget`` (ipympl).
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    H_img, W_img = frame.shape[:2]
    P_arr = np.asarray(initial_P, dtype=np.float64)

    def _project_world_pt(wx: float, wy: float) -> Optional[Tuple[float, float]]:
        p = P_arr @ np.array([
            wx - PITCH_LENGTH / 2, wy - PITCH_WIDTH / 2, 0.0, 1.0,
        ])
        if p[2] <= 0:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    # state[name] = {"image": ((p1x, p1y), (p2x, p2y)), "confirmed": bool}
    # Insertion order is preserved (Python 3.7+) — used for stable colours
    # and "remove last" semantics.
    state: Dict[str, Dict[str, Any]] = {}
    for name, (w1, w2) in NAMED_PITCH_LINES.items():
        i1 = _project_world_pt(*w1)
        i2 = _project_world_pt(*w2)
        if i1 is None or i2 is None:
            continue
        state[name] = {"image": (i1, i2), "confirmed": False}

    # Replay any previously-saved edits from a prior session.
    if existing is not None:
        for entry in existing.line_correspondences or []:
            nm = entry["line_name"]
            if nm not in state:
                continue
            (p1, p2) = entry["image_endpoints"]
            state[nm]["image"] = ((float(p1[0]), float(p1[1])),
                                   (float(p2[0]), float(p2[1])))
            state[nm]["confirmed"] = True

    btn_confirm_all = widgets.Button(
        description="Confirm all", icon="check",
        layout=widgets.Layout(width="auto"),
    )
    btn_reset = widgets.Button(
        description="Reset", icon="undo", button_style="warning",
        layout=widgets.Layout(width="auto"),
    )
    btn_save = widgets.Button(
        description="Save", icon="save", button_style="success",
        layout=widgets.Layout(width="auto"),
    )
    status = widgets.HTML()

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.footer_visible = False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, W_img)
    ax.set_ylim(H_img, 0)
    ax.set_title(f"{slug}  frame {frame_number}  —  click line to confirm, "
                  "drag endpoint to adjust")

    drag_state: Dict[str, Any] = {"line": None, "endpoint": 0}
    LINE_COLORS = plt.cm.tab10.colors

    def _confirmed_corrs() -> List[
        Tuple[str,
              Tuple[Tuple[float, float], Tuple[float, float]],
              Tuple[Tuple[float, float], Tuple[float, float]]]
    ]:
        return [
            (name, st["image"], NAMED_PITCH_LINES[name])
            for name, st in state.items() if st["confirmed"]
        ]

    def _current_h() -> Optional[np.ndarray]:
        corrs = _confirmed_corrs()
        if len(corrs) < 4:
            return None
        return homography_from_line_pairs(
            [c[1] for c in corrs], [c[2] for c in corrs],
        )

    def _redraw():
        # Wipe everything on top of the imshow.
        for ln in list(ax.lines):
            ln.remove()
        for tx in list(ax.texts):
            tx.remove()

        for i, (name, st) in enumerate(state.items()):
            (x0, y0), (x1, y1) = st["image"]
            if st["confirmed"]:
                color = LINE_COLORS[i % len(LINE_COLORS)]
                ax.plot([x0, x1], [y0, y1], "-", color=color,
                        lw=2.5, alpha=0.95)
                ax.plot([x0, x1], [y0, y1], "o", color=color,
                        markersize=9, markeredgecolor="white",
                        markeredgewidth=1.5)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                ax.annotate(name, (mx, my), color=color, fontsize=8,
                            xytext=(5, -5), textcoords="offset points",
                            fontweight="bold")
            else:
                ax.plot([x0, x1], [y0, y1], "-", color="gray",
                        lw=1.0, alpha=0.4)

        # Live H preview, once 4+ lines are confirmed.
        H_fit = _current_h()
        if H_fit is not None:
            try:
                H_inv = np.linalg.inv(H_fit)
            except np.linalg.LinAlgError:
                H_inv = None
            if H_inv is not None:
                for nm, (w1, w2) in NAMED_PITCH_LINES.items():
                    eps = []
                    for w in (w1, w2):
                        wh = H_inv @ np.array([w[0], w[1], 1.0])
                        if abs(wh[2]) < 1e-9:
                            eps = []
                            break
                        eps.append((wh[0] / wh[2], wh[1] / wh[2]))
                    if len(eps) == 2:
                        ax.plot(
                            [eps[0][0], eps[1][0]],
                            [eps[0][1], eps[1][1]],
                            "--", color="cyan", lw=1.0, alpha=0.6,
                        )

        fig.canvas.draw_idle()

    def _update_status():
        n_conf = sum(1 for st in state.values() if st["confirmed"])
        n_total = len(state)
        if n_conf < 4:
            status.value = (
                f"<b>{n_conf}/{n_total}</b> lines confirmed — need at least "
                "4 for a fit. Click on lines that already sit on painted "
                "pitch markings; drag any that are off."
            )
            return
        H_fit = _current_h()
        if H_fit is None:
            status.value = (
                f"<b>{n_conf}</b> lines confirmed — degenerate "
                "configuration. Confirm a line in a different direction."
            )
            return
        corrs = _confirmed_corrs()
        err = line_alignment_error_px(
            [c[1] for c in corrs], [c[2] for c in corrs], H_fit,
        )
        color = "limegreen" if err < 5 else ("orange" if err < 15 else "red")
        status.value = (
            f"<b>{n_conf}/{n_total}</b> lines confirmed — mean line distance: "
            f"<span style='color:{color}'>{err:.1f} px</span>"
        )

    def _find_handle_at(x: float, y: float) -> Optional[Tuple[str, int]]:
        """Nearest confirmed-line endpoint to (x, y) within handle_radius_px."""
        best: Optional[Tuple[str, int]] = None
        best_d2 = handle_radius_px ** 2
        for name, st in state.items():
            if not st["confirmed"]:
                continue
            for ep_idx, (ex, ey) in enumerate(st["image"]):
                d2 = (ex - x) ** 2 + (ey - y) ** 2
                if d2 < best_d2:
                    best = (name, ep_idx)
                    best_d2 = d2
        return best

    def _find_line_near(x: float, y: float) -> Optional[str]:
        """Nearest unconfirmed line to (x, y) within line_pick_px (perpendicular
        distance to the line, ignoring whether the foot of the perpendicular
        lands inside the rendered segment — long lines often extend off-screen)."""
        best: Optional[str] = None
        best_d = line_pick_px
        for name, st in state.items():
            if st["confirmed"]:
                continue
            (x0, y0), (x1, y1) = st["image"]
            dx, dy = x1 - x0, y1 - y0
            seg_len = float(np.hypot(dx, dy))
            if seg_len < 1e-6:
                continue
            d = abs((x - x0) * dy - (y - y0) * dx) / seg_len
            if d < best_d:
                best = name
                best_d = d
        return best

    def _on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        # Confirmed-line handles win over line-clicks.
        h = _find_handle_at(x, y)
        if h is not None:
            drag_state["line"], drag_state["endpoint"] = h
            return
        line = _find_line_near(x, y)
        if line is not None:
            state[line]["confirmed"] = True
            _redraw()
            _update_status()

    def _on_motion(event):
        if drag_state["line"] is None or event.inaxes is not ax:
            return
        if event.xdata is None:
            return
        nm = drag_state["line"]
        ep_idx = drag_state["endpoint"]
        old = state[nm]["image"]
        new_pt = (float(event.xdata), float(event.ydata))
        state[nm]["image"] = (new_pt, old[1]) if ep_idx == 0 else (old[0], new_pt)
        _redraw()

    def _on_release(_event):
        if drag_state["line"] is None:
            return
        drag_state["line"] = None
        _update_status()

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    def _on_confirm_all(_b):
        for st in state.values():
            st["confirmed"] = True
        _redraw()
        _update_status()

    def _on_reset(_b):
        for name, st in state.items():
            i1 = _project_world_pt(*NAMED_PITCH_LINES[name][0])
            i2 = _project_world_pt(*NAMED_PITCH_LINES[name][1])
            if i1 is None or i2 is None:
                continue
            st["image"] = (i1, i2)
            st["confirmed"] = False
        _redraw()
        _update_status()

    def _on_save(_b):
        corrs = _confirmed_corrs()
        if len(corrs) < 4:
            status.value = (
                f"<span style='color:red'>Need at least 4 confirmed lines "
                f"(have {len(corrs)}).</span>"
            )
            return
        H_fit = _current_h()
        if H_fit is None:
            status.value = (
                "<span style='color:red'>Configuration is degenerate — "
                "confirm a line in a different direction.</span>"
            )
            return
        path = save_line_calibration(slug, frame_number, W_img, H_img, corrs)
        err = line_alignment_error_px(
            [c[1] for c in corrs], [c[2] for c in corrs], H_fit,
        )
        status.value = (
            f"<b>Saved</b> {len(corrs)} lines to {path.name}  —  "
            f"alignment error {err:.1f} px."
        )

    btn_confirm_all.on_click(_on_confirm_all)
    btn_reset.on_click(_on_reset)
    btn_save.on_click(_on_save)

    _redraw()
    _update_status()

    return widgets.VBox([
        widgets.HBox([btn_confirm_all, btn_reset, btn_save]),
        status,
    ])


def grab_frame(slug: str, frame_number: int) -> np.ndarray:
    """Convenience helper for the notebook: read a specific frame of a match."""
    video_path = str(Config.MATCH_VIDEOS[slug])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_number} of {slug}")
    return frame
