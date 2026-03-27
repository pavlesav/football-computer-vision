import cv2
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from collections import Counter
from ultralytics import YOLO

from .config import Config
from .segmentation import segment_frame, extract_player_mask, extract_lab_features

_FEAT_DIM = 3  # L, a, b — median Lab color of seg-masked jersey pixels

# Players whose max team-membership probability falls below this threshold
# are classified as Referee / Goalkeeper (unique kit → outlier from both teams).
_REFEREE_CONF_THRESHOLD = 0.75


class TeamClassifier:
    """
    Classify players into teams using seg-masked Lab median jersey color.

    For each detected person:
      1. Run YOLOv8-seg on the full frame to get a person mask.
      2. Slice the jersey area (top 40% height, center 50% width).
      3. Compute the per-pixel median in Lab color space over only the
         masked foreground pixels — grass, ads, and background are excluded.

    Lab is perceptually uniform: L (lightness) separates white from dark kits,
    a (green↔red) and b (blue↔yellow) separate most team colors.
    A 3-dim GMM is stable, interpretable, and handles similar-looking kits
    (e.g. white vs blue-striped) far better than HS histograms.

    Fits k=2 Gaussians for the two outfield kit colors. Anyone whose max
    membership probability falls below _REFEREE_CONF_THRESHOLD is classified
    as Referee/Goalkeeper.
    """

    # Team IDs used for visualization
    TEAM_A = 0
    TEAM_B = 1
    TEAM_REFEREE = 2

    def __init__(self):
        self.gmm = None
        self._fitted = False
        self.cluster_to_team = {}  # gmm cluster index → TEAM_A / TEAM_B
        self.seg_model = YOLO(Config.YOLO_SEG_MODEL)
        self._device = self._resolve_device()
        print(f"TeamClassifier: seg-masked Lab median ({_FEAT_DIM}-dim, "
              f"k=2 GMM + conf-gate referee)")

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

    def _embedding_from_mask(self, frame: np.ndarray, bbox: list,
                              frame_mask: np.ndarray) -> np.ndarray:
        """Extract Lab median for one bbox given a pre-computed frame mask."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h < 8 or w < 8:
            return np.zeros(_FEAT_DIM, dtype=np.float32)

        masked_crop, mask = extract_player_mask(frame, bbox, frame_mask)

        # Focus on jersey area: top 40% height, center 50% width
        jersey_h = max(1, int(h * 0.40))
        margin_w = int(w * 0.25)
        jersey_crop = masked_crop[:jersey_h, margin_w:w - margin_w]
        jersey_mask = mask[:jersey_h, margin_w:w - margin_w]

        if jersey_crop.size == 0:
            return np.zeros(_FEAT_DIM, dtype=np.float32)

        return extract_lab_features(jersey_crop, jersey_mask)

    def extract_embedding(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Extract Lab median from the seg-masked jersey area (single bbox).

        Runs segmentation on the full frame. For multiple bboxes in the same
        frame, prefer :meth:`extract_embeddings_batch` which segments once.
        """
        frame_mask = segment_frame(frame, self.seg_model, device=self._device)
        return self._embedding_from_mask(frame, bbox, frame_mask)

    def extract_embeddings_batch(self, frame: np.ndarray,
                                  bboxes: list) -> np.ndarray:
        """Extract Lab median features for multiple bboxes.

        Runs YOLOv8-seg once on the full frame, then slices each bbox.
        """
        if not bboxes:
            return np.empty((0, _FEAT_DIM), dtype=np.float32)
        frame_mask = segment_frame(frame, self.seg_model, device=self._device)
        return np.array(
            [self._embedding_from_mask(frame, b, frame_mask) for b in bboxes],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Collection / fitting
    # ------------------------------------------------------------------

    def collect_embeddings(self, frame: np.ndarray, players: list) -> list:
        """Collect embeddings for all detected persons on screen."""
        bboxes = [p["bbox"] for p in players]
        if not bboxes:
            return []
        return list(self.extract_embeddings_batch(frame, bboxes))

    def fit(self, all_embeddings: np.ndarray):
        """
        Fit a 2-component GMM on Lab median jersey colors.

        Only the two dominant outfield kit colors are modelled. Outliers
        (goalkeepers, referees, staff) are detected at predict time via the
        confidence threshold — they don't need their own Gaussian.
        """
        # Drop zero vectors (grass/background crops filtered out at extraction)
        valid = np.linalg.norm(all_embeddings, axis=1) > 0
        embeddings = all_embeddings[valid]
        n_samples = len(embeddings)

        self.gmm = GaussianMixture(
            n_components=2,           # exactly 2 team colors
            covariance_type="full",   # full covariance — 3-dim Lab is low-rank enough
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        self.gmm.fit(embeddings)
        self._fitted = True

        # Assign team IDs: larger cluster → Team A (typically home/more players on screen)
        labels = self.gmm.predict(embeddings)
        counter = Counter(labels)
        (big_cluster, big_n), (small_cluster, small_n) = counter.most_common()
        self.cluster_to_team = {big_cluster: self.TEAM_A, small_cluster: self.TEAM_B}

        # Estimate how many are referee/GK (low-confidence predictions)
        probs = self.gmm.predict_proba(embeddings)
        n_outliers = int((probs.max(axis=1) < _REFEREE_CONF_THRESHOLD).sum())

        print(f"Team classification fitted on {n_samples} seg-masked Lab colors ({_FEAT_DIM}-dim)")
        print(f"  GMM weights  : Team A={self.gmm.weights_[big_cluster]:.1%}, "
              f"Team B={self.gmm.weights_[small_cluster]:.1%}")
        print(f"  Cluster sizes: Team A={big_n}, Team B={small_n}")
        print(f"  Outliers (conf < {_REFEREE_CONF_THRESHOLD:.0%}): ~{n_outliers} "
              f"({n_outliers/n_samples:.1%}) → classified as Referee/GK")

    def predict(self, embedding: np.ndarray) -> int:
        """Predict team ID for a single Lab color vector.

        Returns TEAM_REFEREE for anyone who doesn't confidently match either
        team Gaussian (goalkeepers, referees, staff, ball-boys).
        """
        if not self._fitted:
            raise RuntimeError("TeamClassifier not fitted yet. Call fit() first.")
        if np.linalg.norm(embedding) == 0:
            return self.TEAM_REFEREE
        probs = self.gmm.predict_proba(embedding.reshape(1, -1))[0]
        if probs.max() < _REFEREE_CONF_THRESHOLD:
            return self.TEAM_REFEREE
        cluster = int(probs.argmax())
        return self.cluster_to_team[cluster]

    # ------------------------------------------------------------------
    # Per-frame classification
    # ------------------------------------------------------------------

    def classify_players(self, frame: np.ndarray, players: list) -> list:
        """
        Add ``'team_id'`` to each player detection dict.

        Each player is segmented, Lab median is extracted from the
        jersey area, and the GMM predicts the most likely team cluster.
        Outliers (goalkeepers, referees) are flagged via confidence gating.
        """
        if not players:
            return players

        bboxes = [p["bbox"] for p in players]
        embeddings = self.extract_embeddings_batch(frame, bboxes)
        for p, emb in zip(players, embeddings):
            p["team_id"] = self.predict(emb)

        return players
