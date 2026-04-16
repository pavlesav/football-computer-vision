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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

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
    correspondences: List[Tuple[str, Tuple[float, float], Tuple[float, float]]]
    H_image_to_pitch_natural: Optional[List[List[float]]] = None
    P_pnlcalib_convention: Optional[List[List[float]]] = None
    reprojection_error_m: Optional[float] = None


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
    return ManualCalibration(**raw)


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
