"""
Optical-flow camera motion tracker for pitch homography.

Purpose
-------
PnLCalib is frame-independent: each frame is processed from scratch. On
Tier-3 matches (night/dusk, oblique cameras, running tracks) it returns
None for 50-90% of frames, leaving large gaps where the pipeline falls
back to the stale smoothed projection.

This module fills those gaps by tracking camera motion between frames:

  1. When PnLCalib delivers a trusted projection P, we seed the tracker.
     Good features are detected on the image (cv2.goodFeaturesToTrack)
     and the inverse of the plane homography gives us the pitch-world
     coordinate of every feature.

  2. On subsequent frames we run Lucas-Kanade optical flow to track those
     features. Each survived feature is a (new_image_pt, world_pt) pair,
     so we can re-estimate the image->pitch homography with RANSAC.

  3. When PnLCalib succeeds again, we re-seed and discard the old trail.

We track the 3x3 plane homography (image -> pitch, z=0) rather than the
full 3x4 projection. Tracked features live on the pitch plane by
construction, so PnP with coplanar points would be degenerate. The 3x3
homography is all we need for player foot -> pitch position, which is
what the downstream event pipeline consumes.

For rendering (src/run_pnlcalib_video.py::project_lines) we pack the
3x3 into a 3x4 with a zero z-column. This draws every z=0 line
correctly (halfway line, penalty boxes, touchlines, center circle) and
drops the goal-post bars — an acceptable trade for filling Tier-3 gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


_MAX_COND = 1e12   # beyond this a homography is numerically degenerate


def _safe_inv(H: np.ndarray) -> Optional[np.ndarray]:
    """Invert a 3x3 homography, or None if it is singular/degenerate.

    cv2.findHomography (RANSAC) can return a rank-deficient H when the
    surviving inliers are nearly collinear; inverting it unguarded killed a
    batch half at 87% (jez-ars p2, 2026-07-04) after ~500k clean frames.
    Callers must treat None as "no usable homography this frame".
    """
    try:
        if not np.all(np.isfinite(H)) or np.linalg.cond(H) > _MAX_COND:
            return None
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def _h_from_projection(P: np.ndarray) -> Optional[np.ndarray]:
    """Extract the 3x3 image->pitch plane homography from a 3x4 P matrix.

    P operates on origin-centered pitch coordinates (PnLCalib convention),
    so the returned H also maps image pixels to origin-centered pitch
    coords (x in [-52.5, 52.5], y in [-34, 34]). None if P's plane part is
    degenerate.
    """
    H_pitch_to_img = P[:, [0, 1, 3]]
    return _safe_inv(H_pitch_to_img)


def projection_from_plane_homography(
        H_img_to_pitch: np.ndarray) -> Optional[np.ndarray]:
    """Pack a 3x3 plane homography into a 3x4 projection matrix with zero z-column.

    The resulting P is only correct for z=0 world points. Goal posts
    (z != 0) won't render; every pitch line renders correctly. None if the
    homography is degenerate (not invertible).
    """
    H_pitch_to_img = _safe_inv(H_img_to_pitch)
    if H_pitch_to_img is None:
        return None
    P = np.zeros((3, 4), dtype=np.float64)
    P[:, 0] = H_pitch_to_img[:, 0]
    P[:, 1] = H_pitch_to_img[:, 1]
    P[:, 3] = H_pitch_to_img[:, 2]
    return P


@dataclass
class _TrackerState:
    prev_gray: np.ndarray
    img_pts: np.ndarray            # Nx1x2 float32 (image pixels)
    world_pts: np.ndarray          # Nx2 float32 (origin-centered pitch coords)
    H_img_to_pitch: np.ndarray     # 3x3 homography at time of most recent update
    frames_since_seed: int = 0


class CameraMotionTracker:
    """Track the image->pitch homography across frames via optical flow.

    Usage
    -----
    tracker = CameraMotionTracker()
    for frame in frames:
        if pnlcalib_gave_trusted_P:
            P = tracker.seed(frame, trusted_P)         # returns trusted_P unchanged
        else:
            P = tracker.propagate(frame)               # returns 3x4 P or None

    Parameters
    ----------
    max_features       : upper bound on features per seed (more = slower).
    min_features       : below this, propagation gives up and returns None.
    reseed_interval    : force a re-seed after this many propagated frames
                         to prevent drift from accumulating indefinitely.
    lk_win_size        : Lucas-Kanade window; larger is more robust to
                         blur and big motion, slower.
    ransac_thresh_px   : RANSAC reprojection threshold for the new plane
                         homography, in image pixels.
    margin_m           : world points within this margin (in metres) of
                         the pitch rectangle are retained; others are
                         features on stands/logos and get dropped.
    """

    def __init__(
        self,
        max_features: int = 300,
        min_features: int = 20,
        reseed_interval: int = 60,
        lk_win_size: int = 21,
        lk_max_level: int = 3,
        ransac_thresh_px: float = 3.0,
        margin_m: float = 5.0,
        feature_quality: float = 0.01,
        feature_min_distance: int = 10,
    ):
        self.max_features = max_features
        self.min_features = min_features
        self.reseed_interval = reseed_interval
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.ransac_thresh_px = ransac_thresh_px
        self.margin_m = margin_m
        self.feature_quality = feature_quality
        self.feature_min_distance = feature_min_distance

        self._state: Optional[_TrackerState] = None

        # Bookkeeping for diagnostics
        self.seeds = 0
        self.propagations = 0
        self.propagation_failures = 0

    # ── Public API ──────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._state is not None

    def reset(self) -> None:
        """Forget everything (e.g. after a confirmed camera cut)."""
        self._state = None

    def seed(self, frame: np.ndarray, P_trusted: np.ndarray) -> np.ndarray:
        """Feed a trusted projection matrix and reset the feature trail.

        Returns the same P_trusted (for call-site symmetry with propagate).
        If feature detection fails, tracker is cleared and P_trusted is
        still returned — next call to propagate() will simply return None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_img_to_pitch = _h_from_projection(P_trusted)
        if H_img_to_pitch is None:
            # Degenerate trusted P — don't build a trail on it.
            self._state = None
            return P_trusted
        ok = self._detect_and_store(gray, H_img_to_pitch)
        if ok:
            self.seeds += 1
        return P_trusted

    def propagate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate P for the new frame via optical flow + RANSAC homography.

        Returns a 3x4 P (z-column zero) or None if the trail was lost.
        """
        if self._state is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = self._state

        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            s.prev_gray,
            gray,
            s.img_pts,
            None,
            winSize=(self.lk_win_size, self.lk_win_size),
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if new_pts is None or status is None:
            self._state = None
            self.propagation_failures += 1
            return None

        mask = status.flatten().astype(bool)
        if mask.sum() < self.min_features:
            self._state = None
            self.propagation_failures += 1
            return None

        tracked_img = new_pts[mask].reshape(-1, 2).astype(np.float32)
        tracked_world = s.world_pts[mask]

        H_img_to_pitch_new, inliers = cv2.findHomography(
            tracked_img,
            tracked_world,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh_px,
        )
        if H_img_to_pitch_new is None or inliers is None:
            self._state = None
            self.propagation_failures += 1
            return None

        inlier_mask = inliers.flatten().astype(bool)
        if inlier_mask.sum() < self.min_features:
            self._state = None
            self.propagation_failures += 1
            return None

        # Keep only RANSAC inliers — prevents drifting tracks from poisoning
        # the next iteration's homography estimate.
        kept_img = tracked_img[inlier_mask]
        kept_world = tracked_world[inlier_mask]

        s.prev_gray = gray
        s.img_pts = kept_img.reshape(-1, 1, 2).astype(np.float32)
        s.world_pts = kept_world
        s.H_img_to_pitch = H_img_to_pitch_new
        s.frames_since_seed += 1

        # Drift guard: periodically re-detect features using the current
        # homography estimate. This replaces drifting tracks with fresh ones.
        if s.frames_since_seed >= self.reseed_interval:
            refreshed = self._detect_and_store(gray, H_img_to_pitch_new)
            if not refreshed:
                # Keep the old trail; detection can fail transiently on
                # motion-blurred frames.
                s.frames_since_seed = 0

        P_out = projection_from_plane_homography(s.H_img_to_pitch)
        if P_out is None:
            # Flow RANSAC produced a degenerate homography — trail is junk.
            self._state = None
            return None
        self.propagations += 1
        return P_out

    # ── Internal ────────────────────────────────────────────────────────

    def _detect_and_store(
        self, gray: np.ndarray, H_img_to_pitch: np.ndarray
    ) -> bool:
        """Detect good features on the pitch, assign world coords, store state."""
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.feature_quality,
            minDistance=self.feature_min_distance,
            blockSize=7,
        )
        if features is None or len(features) < self.min_features:
            return False

        # Project each feature into pitch coordinates. Drop anything that
        # lands far outside the pitch — those are stands, logos, crowd
        # which will violate the planar assumption.
        img_xy = features.reshape(-1, 2).astype(np.float64)
        ones = np.ones((img_xy.shape[0], 1))
        img_h = np.hstack([img_xy, ones])                 # Nx3
        world_h = img_h @ H_img_to_pitch.T                # Nx3

        z = world_h[:, 2]
        valid = np.abs(z) > 1e-10
        wx = np.where(valid, world_h[:, 0] / np.where(valid, z, 1.0), 0.0)
        wy = np.where(valid, world_h[:, 1] / np.where(valid, z, 1.0), 0.0)

        lim_x = PITCH_LENGTH / 2 + self.margin_m
        lim_y = PITCH_WIDTH / 2 + self.margin_m
        on_pitch = valid & (np.abs(wx) <= lim_x) & (np.abs(wy) <= lim_y)

        if on_pitch.sum() < self.min_features:
            return False

        kept_img = img_xy[on_pitch].astype(np.float32)
        kept_world = np.stack([wx[on_pitch], wy[on_pitch]], axis=1).astype(np.float32)

        self._state = _TrackerState(
            prev_gray=gray,
            img_pts=kept_img.reshape(-1, 1, 2),
            world_pts=kept_world,
            H_img_to_pitch=H_img_to_pitch.astype(np.float64),
            frames_since_seed=0,
        )
        return True
