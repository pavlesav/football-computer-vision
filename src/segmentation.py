"""Player segmentation and feature extraction for team classification.

Uses YOLOv8-seg on the **full frame** to produce per-player masks, then
extracts features from only the masked (foreground) pixels — removing
grass, ads, and other background noise that degrades jersey-color clustering.

Running seg on the full frame (not individual crops) is critical: the model
needs to see full person silhouettes, not isolated jersey fragments.

CNN embedding extraction uses a frozen ResNet18 backbone (pretrained on
ImageNet) to produce a 512-dim feature vector per masked player crop.
This captures texture, pattern, and color information that Lab statistics miss.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO

# ── CNN embedding model (lazy-loaded singleton) ──────────────────────────
_cnn_model = None
_cnn_device = None
_cnn_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),   # height x width — tall crop for full-body players
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _get_cnn(device=None):
    """Return the shared frozen ResNet18 backbone (loads once)."""
    global _cnn_model, _cnn_device
    if _cnn_model is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _cnn_device = torch.device(device) if isinstance(device, str) else torch.device(f"cuda:{device}")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final FC layer — use avgpool output (512-dim)
        model.fc = torch.nn.Identity()
        model.eval()
        model.to(_cnn_device)
        for p in model.parameters():
            p.requires_grad_(False)
        _cnn_model = model
    return _cnn_model, _cnn_device


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
    Extract Lab color statistics from masked player pixels.

    Returns a 6-dim vector: ``[median_L, median_a, median_b, std_L, std_a,
    std_b]``.  Median is robust for central jersey color; std captures
    pattern variation (solid vs striped kits) — giving the GMM an extra
    axis to separate teams whose median colors overlap.

    Args:
        crop: Player crop image (BGR).
        mask: Binary mask (``255`` = include, ``0`` = exclude).

    Returns:
        6-dim vector (float32).  Zero vector if no valid pixels.
    """
    if crop.size == 0 or mask.sum() == 0:
        return np.zeros(6, dtype=np.float32)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    fg = lab[mask > 0]
    median = np.median(fg, axis=0)
    std = np.std(fg, axis=0)
    return np.concatenate([median, std]).astype(np.float32)


def _fill_background_with_mean(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace background pixels with the mean foreground color.

    This prevents large black regions from dominating CNN embeddings.
    """
    fg_pixels = crop[mask > 0]
    if len(fg_pixels) == 0:
        return crop.copy()
    mean_color = fg_pixels.mean(axis=0).astype(np.uint8)
    result = crop.copy()
    result[mask == 0] = mean_color
    return result


def extract_cnn_embedding(crop: np.ndarray, mask: np.ndarray,
                          device=None) -> np.ndarray:
    """
    Extract a 512-dim CNN embedding from a masked player crop.

    Uses a frozen ResNet18 backbone (pretrained ImageNet).  Background
    pixels are filled with the mean foreground color so they don't
    bias the embedding.

    Args:
        crop: Player crop image (BGR).
        mask: Binary mask (``255`` = include, ``0`` = exclude).
        device: Inference device (int for GPU index, ``'cpu'``, etc.).

    Returns:
        512-dim vector (float32).  Zero vector if crop is invalid.
    """
    if crop.size == 0 or mask.sum() == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return np.zeros(512, dtype=np.float32)

    model, dev = _get_cnn(device)

    filled = _fill_background_with_mean(crop, mask)
    rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

    tensor = _cnn_transform(rgb).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = model(tensor).squeeze(0).cpu().numpy()

    return emb.astype(np.float32)


def extract_cnn_embeddings_batch(crops: list, masks: list,
                                 device=None) -> np.ndarray:
    """
    Batched CNN embedding extraction for multiple player crops.

    Args:
        crops: List of player crop images (BGR).
        masks: List of binary masks corresponding to each crop.
        device: Inference device.

    Returns:
        Array of shape ``(N, 512)`` (float32).
    """
    if not crops:
        return np.empty((0, 512), dtype=np.float32)

    model, dev = _get_cnn(device)

    tensors = []
    valid_indices = []
    for i, (crop, mask) in enumerate(zip(crops, masks)):
        if crop.size == 0 or mask.sum() == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        filled = _fill_background_with_mean(crop, mask)
        rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
        tensors.append(_cnn_transform(rgb))
        valid_indices.append(i)

    result = np.zeros((len(crops), 512), dtype=np.float32)

    if tensors:
        batch = torch.stack(tensors).to(dev)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        for j, idx in enumerate(valid_indices):
            result[idx] = embeddings[j]

    return result
