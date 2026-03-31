import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from collections import Counter
from ultralytics import YOLO

from .config import Config
from .segmentation import (segment_frame, extract_player_mask,
                           extract_cnn_embeddings_batch)

_CNN_DIM = 512     # raw ResNet18 embedding size
_PCA_DIM = 32      # reduced dimension for GMM clustering


class TeamClassifier:
    """
    Classify players into teams using CNN embeddings from masked crops.

    For each detected person:
      1. Run YOLOv8-seg on the full frame to get a person mask.
      2. Apply the mask to the player crop (background zeroed).
      3. Extract a 512-dim embedding via frozen ResNet18 backbone.

    PCA reduces to 32 dims, then a two-pass k=2 GMM clusters into teams.
    CNN features capture texture, pattern, logos, and color simultaneously
    — separating kits that look similar in raw color statistics.
    """

    TEAM_A = 0
    TEAM_B = 1

    def __init__(self):
        self.gmm = None
        self.pca = None
        self._fitted = False
        self.cluster_to_team = {}  # gmm cluster index → TEAM_A / TEAM_B
        self.seg_model = YOLO(Config.YOLO_SEG_MODEL)
        self._device = self._resolve_device()
        print(f"TeamClassifier: ResNet18 CNN embeddings ({_CNN_DIM}→{_PCA_DIM}-dim PCA, k=2 GMM)")

    @staticmethod
    def _resolve_device():
        if not torch.cuda.is_available():
            return "cpu"
        try:
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            if f"sm_{cc_major}{cc_minor}" in set(torch.cuda.get_arch_list()):
                return 0
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _crops_and_masks(self, frame: np.ndarray, bboxes: list,
                         frame_mask: np.ndarray) -> tuple:
        """Extract masked crops for a list of bboxes given a frame mask."""
        crops, masks = [], []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = y2 - y1, x2 - x1
            if h < 8 or w < 8:
                crops.append(np.zeros((8, 8, 3), dtype=np.uint8))
                masks.append(np.zeros((8, 8), dtype=np.uint8))
                continue
            masked_crop, mask = extract_player_mask(frame, bbox, frame_mask)
            crops.append(masked_crop)
            masks.append(mask)
        return crops, masks

    def extract_embeddings_batch(self, frame: np.ndarray,
                                 bboxes: list) -> np.ndarray:
        """Extract CNN embeddings for multiple bboxes.

        Runs YOLOv8-seg once on the full frame, then feeds each masked
        crop through ResNet18 in a single batch.
        """
        if not bboxes:
            return np.empty((0, _CNN_DIM), dtype=np.float32)
        frame_mask = segment_frame(frame, self.seg_model, device=self._device)
        crops, masks = self._crops_and_masks(frame, bboxes, frame_mask)
        return extract_cnn_embeddings_batch(crops, masks, device=self._device)

    # ------------------------------------------------------------------
    # Collection / fitting
    # ------------------------------------------------------------------

    def collect_embeddings(self, frame: np.ndarray, players: list) -> list:
        """Collect CNN embeddings for all detected persons on screen."""
        bboxes = [p["bbox"] for p in players]
        if not bboxes:
            return []
        return list(self.extract_embeddings_batch(frame, bboxes))

    def fit(self, all_embeddings: np.ndarray, outlier_percentile: float = 5.0):
        """
        PCA + two-pass GMM fit.

        1. PCA from 512 → 32 dims (removes noise, makes GMM stable).
        2. Pass 1: rough k=2 GMM on all data.
        3. Pass 2: remove bottom ``outlier_percentile`` by membership
           probability, re-fit for tighter centroids.
        """
        # Drop zero vectors (failed crops)
        valid = np.linalg.norm(all_embeddings, axis=1) > 0
        embeddings = all_embeddings[valid]
        n_total = len(embeddings)

        # ── PCA ───────────────────────────────────────────────────────────
        n_components = min(_PCA_DIM, embeddings.shape[0], embeddings.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        variance_kept = self.pca.explained_variance_ratio_.sum()

        gmm_kwargs = dict(
            n_components=2,
            covariance_type="full",
            n_init=10,
            max_iter=300,
            random_state=42,
        )

        # ── Pass 1: rough fit ─────────────────────────────────────────────
        rough_gmm = GaussianMixture(**gmm_kwargs)
        rough_gmm.fit(reduced)

        probs = rough_gmm.predict_proba(reduced)
        max_probs = probs.max(axis=1)
        threshold = np.percentile(max_probs, outlier_percentile)
        inlier_mask = max_probs >= threshold
        n_removed = int((~inlier_mask).sum())

        # ── Pass 2: re-fit on inliers only ─────────────────────────────────
        clean = reduced[inlier_mask]

        self.gmm = GaussianMixture(**gmm_kwargs)
        self.gmm.fit(clean)
        self._fitted = True

        # Assign team IDs: larger cluster → Team A
        labels = self.gmm.predict(clean)
        counter = Counter(labels)
        (big_cluster, big_n), (small_cluster, small_n) = counter.most_common()
        self.cluster_to_team = {big_cluster: self.TEAM_A, small_cluster: self.TEAM_B}

        print(f"Team classification fitted on {n_total} samples "
              f"({_CNN_DIM}→{n_components}-dim PCA, {variance_kept:.1%} variance)")
        print(f"  Outliers removed: {n_removed} ({n_removed/n_total:.1%}) — "
              f"GK/ref/staff (bottom {outlier_percentile:.0f}% membership probability)")
        print(f"  GMM weights  : Team A={self.gmm.weights_[big_cluster]:.1%}, "
              f"Team B={self.gmm.weights_[small_cluster]:.1%}")
        print(f"  Cluster sizes: Team A={big_n}, Team B={small_n}")

    def predict(self, embedding: np.ndarray) -> int:
        """Predict team ID for a single CNN embedding."""
        if not self._fitted:
            raise RuntimeError("TeamClassifier not fitted yet. Call fit() first.")
        if np.linalg.norm(embedding) == 0:
            return self.TEAM_A
        reduced = self.pca.transform(embedding.reshape(1, -1))
        probs = self.gmm.predict_proba(reduced)[0]
        cluster = int(probs.argmax())
        return self.cluster_to_team[cluster]

    # ------------------------------------------------------------------
    # Per-frame classification
    # ------------------------------------------------------------------

    def classify_players(self, frame: np.ndarray, players: list) -> list:
        """
        Add ``'team_id'`` to each player detection dict.

        Each player is segmented, a CNN embedding is extracted from the
        masked crop, and the GMM predicts the most likely team cluster.
        """
        if not players:
            return players

        bboxes = [p["bbox"] for p in players]
        embeddings = self.extract_embeddings_batch(frame, bboxes)
        for p, emb in zip(players, embeddings):
            p["team_id"] = self.predict(emb)

        return players
