"""
Manual ball-position annotations.

YOLO loses the ball reliably during dribbles, occlusions, and when it sits
still in the grass. For a polished demo we let the user click the ball every
N frames in the rendered window, save those clicks to disk, and have the
renderer linearly interpolate between them. The result is a ball overlay
that's continuous and locked to the actual ball position regardless of
model accuracy.

On-disk schema: ``data/ball_annotations/{slug}.json``::

    {
      "slug": "sut-mla",
      "annotations": {
        "97200": [950, 540],
        "97205": [965, 545],
        ...
      }
    }

Keys are absolute video frame indices (string-encoded for JSON), values are
``[x, y]`` in image pixels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import Config


def annotation_dir() -> Path:
    d = Config.PROJECT_ROOT / "data" / "ball_annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def annotation_path(slug: str) -> Path:
    return annotation_dir() / f"{slug}.json"


def load_ball_annotations(slug: str) -> Dict[int, Tuple[float, float]]:
    """Return ``{frame_number: (x, y)}`` for the slug, empty dict if no file."""
    path = annotation_path(slug)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        int(k): (float(v[0]), float(v[1]))
        for k, v in raw.get("annotations", {}).items()
    }


def save_ball_annotations(
    slug: str, annotations: Dict[int, Tuple[float, float]],
) -> Path:
    path = annotation_path(slug)
    data = {
        "slug": slug,
        "annotations": {
            str(k): [float(v[0]), float(v[1])]
            for k, v in sorted(annotations.items())
        },
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def interpolate_ball(
    annotations: Dict[int, Tuple[float, float]],
    abs_frame: int,
    max_gap_frames: int = 15,
) -> Optional[Tuple[float, float]]:
    """Linearly interpolate the ball position at ``abs_frame``.

    Returns ``None`` when:
      * there are no annotations at all,
      * ``abs_frame`` is outside the [first_annotation, last_annotation] range,
      * or the bracketing annotations are more than ``max_gap_frames`` apart
        (treats that as "ball wasn't visible — don't fabricate a position").
    """
    if not annotations:
        return None
    if abs_frame in annotations:
        return annotations[abs_frame]
    sorted_frames = sorted(annotations.keys())
    if abs_frame < sorted_frames[0] or abs_frame > sorted_frames[-1]:
        return None
    after_idx = next(i for i, f in enumerate(sorted_frames) if f > abs_frame)
    f1 = sorted_frames[after_idx - 1]
    f2 = sorted_frames[after_idx]
    if f2 - f1 > max_gap_frames:
        return None
    p1 = annotations[f1]
    p2 = annotations[f2]
    t = (abs_frame - f1) / (f2 - f1)
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


# ── Widget ─────────────────────────────────────────────────────────────

def build_ball_label_widget(
    slug: str,
    video_path: str,
    start_frame: int,
    end_frame: int,
    step: int = 5,
):
    """Click-the-ball labeling widget.

    Steps through every ``step``-th frame of the video in
    ``[start_frame, end_frame]``. The user clicks the ball; the click is
    recorded and saved to ``data/ball_annotations/{slug}.json`` immediately,
    then the navigator auto-advances. Existing annotations load on first
    render so this is incremental — close the notebook, re-open the cell,
    pick up where you left off.

    Buttons:
      * ◀ Prev / Next ▶ — navigate
      * Skip — advance without recording (use when the ball isn't visible)
      * Delete — drop the current frame's annotation
      * Save — explicit save (auto-saves on click anyway)

    Requires ``%matplotlib widget``.
    """
    import cv2
    import numpy as np
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    annotations = load_ball_annotations(slug)
    target_frames = list(range(int(start_frame), int(end_frame) + 1, int(step)))
    if not target_frames:
        raise ValueError("Empty frame range")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    state = {"idx": 0}

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(15, 8))
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.footer_visible = False

    img_artist = ax.imshow(np.zeros((H, W, 3), dtype=np.uint8))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    status = widgets.HTML()
    btn_prev = widgets.Button(description="◀ Prev",
                               layout=widgets.Layout(width="auto"))
    btn_next = widgets.Button(description="Next ▶",
                               layout=widgets.Layout(width="auto"))
    btn_skip = widgets.Button(description="Skip (no ball)",
                               layout=widgets.Layout(width="auto"))
    btn_delete = widgets.Button(description="Delete",
                                 button_style="warning",
                                 layout=widgets.Layout(width="auto"))
    btn_save = widgets.Button(description="Save",
                               button_style="success",
                               layout=widgets.Layout(width="auto"))

    def _render():
        f = target_frames[state["idx"]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Wipe previous markers
        for ln in list(ax.lines):
            ln.remove()
        for tx in list(ax.texts):
            tx.remove()
        # Current annotation marker
        if f in annotations:
            x, y = annotations[f]
            ax.plot(x, y, "o", color="lime", markersize=14,
                    markeredgecolor="black", markeredgewidth=2)
        labeled = sum(1 for fr in target_frames if fr in annotations)
        ax.set_title(f"{slug}  frame {f}  "
                     f"({state['idx'] + 1}/{len(target_frames)}, "
                     f"{labeled} labeled)  —  click on the ball")
        status.value = (
            f"<b>{labeled}/{len(target_frames)}</b> frames labeled "
            f"&nbsp;|&nbsp; current frame: <b>{f}</b> "
            f"&nbsp;|&nbsp; "
            f"{'<span style=\"color:limegreen\">labeled</span>' if f in annotations else '<span style=\"color:#bbb\">unlabeled</span>'}"
        )
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        f = target_frames[state["idx"]]
        annotations[f] = (float(event.xdata), float(event.ydata))
        save_ball_annotations(slug, annotations)
        # Auto-advance for fast labeling.
        state["idx"] = min(len(target_frames) - 1, state["idx"] + 1)
        _render()

    fig.canvas.mpl_connect("button_press_event", _on_click)

    def _go(delta):
        state["idx"] = max(0, min(len(target_frames) - 1, state["idx"] + delta))
        _render()

    btn_prev.on_click(lambda _: _go(-1))
    btn_next.on_click(lambda _: _go(1))
    btn_skip.on_click(lambda _: _go(1))

    def _on_delete(_b):
        f = target_frames[state["idx"]]
        if f in annotations:
            del annotations[f]
            save_ball_annotations(slug, annotations)
            _render()

    btn_delete.on_click(_on_delete)
    btn_save.on_click(lambda _: save_ball_annotations(slug, annotations))

    _render()

    return widgets.VBox([
        fig.canvas,
        widgets.HBox([btn_prev, btn_next, btn_skip, btn_delete, btn_save]),
        status,
    ])
