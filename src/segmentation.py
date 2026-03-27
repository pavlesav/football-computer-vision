"""Player segmentation and color feature extraction for team classification.

Uses YOLOv8-seg on the **full frame** to produce per-player masks, then
extracts color features from only the masked (foreground) pixels — removing
grass, ads, and other background noise that degrades jersey-color clustering.

Running seg on the full frame (not individual crops) is critical: the model
needs to see full person silhouettes, not isolated jersey fragments.
"""

import cv2
import numpy as np
from ultralytics import YOLO


def segment_frame(frame: np.ndarray, seg_model: YOLO,
                  device=None) -> np.ndarray:
    """
    Run YOLOv8-seg on the full frame and return a combined person mask.

    Args:
        frame: Full video frame (BGR).
        seg_model: Loaded YOLOv8-seg model instance.
        device: Inference device (0 for GPU, ``'cpu'``, etc.).

    Returns:
        Full-frame binary mask (``uint8``, ``255`` = person pixel,
        ``0`` = background).  All detected persons are merged into one mask.
    """
    fh, fw = frame.shape[:2]

    kwargs = dict(conf=0.25, verbose=False)
    if device is not None:
        kwargs["device"] = device

    results = seg_model(frame, **kwargs)[0]

    mask = np.zeros((fh, fw), dtype=np.uint8)

    if results.masks is not None:
        for i, cls_id in enumerate(results.boxes.cls):
            if int(cls_id) == 0:  # person class in COCO
                seg_data = results.masks.data[i].cpu().numpy()
                seg_resized = cv2.resize(seg_data, (fw, fh),
                                         interpolation=cv2.INTER_LINEAR)
                mask = np.maximum(
                    mask, (seg_resized > 0.5).astype(np.uint8) * 255
                )

    return mask


def extract_player_mask(frame: np.ndarray, bbox: list,
                        frame_mask: np.ndarray) -> tuple:
    """
    Extract a single player's masked crop from a pre-computed frame mask.

    Args:
        frame: Full video frame (BGR).
        bbox: ``[x1, y1, x2, y2]`` bounding box (pixel coords).
        frame_mask: Full-frame person mask from :func:`segment_frame`.

    Returns:
        masked_crop: Player crop with background zeroed out (BGR).
        mask: Binary crop-mask (``255`` = player, ``0`` = background).
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = frame_mask[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]

    # Fallback: if no person pixels inside this bbox, use whole crop
    if crop_mask.sum() == 0:
        crop_mask = np.ones((h, w), dtype=np.uint8) * 255

    masked_crop = cv2.bitwise_and(crop, crop, mask=crop_mask)
    return masked_crop, crop_mask


def segment_player(frame: np.ndarray, bbox: list,
                   seg_model: YOLO, device=None) -> tuple:
    """
    Convenience wrapper: segment *one* player from a full frame.

    Runs :func:`segment_frame` on the whole image, then slices the bbox
    region.  For batch usage (multiple players per frame), call
    :func:`segment_frame` once and :func:`extract_player_mask` per bbox.

    Returns:
        masked_crop, mask — same as :func:`extract_player_mask`.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 8:
        crop = frame[y1:y2, x1:x2].copy()
        return crop, np.ones((h, w), dtype=np.uint8) * 255

    frame_mask = segment_frame(frame, seg_model, device=device)
    return extract_player_mask(frame, bbox, frame_mask)


def extract_hsv_features(crop: np.ndarray, mask: np.ndarray,
                         h_bins: int = 8, s_bins: int = 4) -> np.ndarray:
    """
    Extract a normalized 2D Hue–Saturation histogram from masked player pixels.

    Only pixels where ``mask > 0`` contribute to the histogram, so background
    grass and other noise are excluded.

    Args:
        crop: Player crop image (BGR).
        mask: Binary mask (``255`` = include, ``0`` = exclude).
        h_bins: Number of bins for Hue channel (range 0–180 in OpenCV).
        s_bins: Number of bins for Saturation channel (range 0–256).

    Returns:
        Flattened, L1-normalized histogram of shape ``(h_bins * s_bins,)``.
    """
    n_features = h_bins * s_bins

    if crop.size == 0 or mask.sum() == 0:
        return np.zeros(n_features, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], mask,
        [h_bins, s_bins],
        [0, 180, 0, 256],
    )

    total = hist.sum()
    if total > 0:
        hist /= total

    return hist.flatten().astype(np.float32)


def extract_lab_features(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract the median Lab color from masked player pixels.

    Lab is perceptually uniform — ``L`` captures brightness (white vs dark),
    ``a`` separates green↔red, ``b`` separates blue↔yellow.  The median is
    robust to small segmentation artefacts and mixed-color stripes.

    Args:
        crop: Player crop image (BGR).
        mask: Binary mask (``255`` = include, ``0`` = exclude).

    Returns:
        3-dim vector ``[L, a, b]`` (float32).  Zero vector if no valid pixels.
    """
    if crop.size == 0 or mask.sum() == 0:
        return np.zeros(3, dtype=np.float32)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    return np.median(lab[mask > 0], axis=0).astype(np.float32)
