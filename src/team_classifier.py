import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.cluster import KMeans
from collections import Counter

from .config import Config

_CROP_SIZE = 128  # input size for ResNet feature extraction
_EMBED_DIM = 512  # ResNet18 output dimensionality


class TeamClassifier:
    """
    Classify players into teams using ResNet18 appearance embeddings.

    Extracts a 512-dim feature vector from each player crop via a pretrained
    ResNet18 backbone, then clusters field players into 2 teams with KMeans.
    YOLO class IDs separate referees/goalkeepers before clustering.
    """

    # Fine-tuned model class IDs
    CLS_PLAYER = 0
    CLS_GOALKEEPER = 2
    CLS_REFEREE = 3

    # Team IDs used for visualization
    TEAM_A = 0
    TEAM_B = 1
    TEAM_REFEREE = 2

    def __init__(self):
        self.n_clusters = 2
        self.kmeans = None
        self._fitted = False
        self.cluster_to_team = {}

        # ResNet18 feature extractor (pretrained, frozen, no FC)
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        backbone.eval()
        self._backbone = backbone

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((_CROP_SIZE, _CROP_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print(f"TeamClassifier: ResNet18 backbone loaded ({_EMBED_DIM}-dim embeddings)")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _crop_player(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Extract the player crop (full bbox) as an RGB array."""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def extract_embedding(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Extract a single 512-dim embedding for one player crop."""
        crop_rgb = self._crop_player(frame, bbox)
        tensor = self._transform(crop_rgb).unsqueeze(0)
        with torch.no_grad():
            emb = self._backbone(tensor).squeeze(0).numpy()
        return emb

    def extract_embeddings_batch(self, frame: np.ndarray,
                                  bboxes: list) -> np.ndarray:
        """Extract embeddings for multiple bboxes in a single forward pass."""
        if not bboxes:
            return np.empty((0, _EMBED_DIM), dtype=np.float32)
        tensors = [self._transform(self._crop_player(frame, b)) for b in bboxes]
        batch = torch.stack(tensors)
        with torch.no_grad():
            embeddings = self._backbone(batch).numpy()
        return embeddings

    # ------------------------------------------------------------------
    # Collection / fitting
    # ------------------------------------------------------------------

    def collect_embeddings(self, frame: np.ndarray, players: list) -> list:
        """Collect embeddings for field players only (skip referees/goalkeepers)."""
        field_bboxes = [
            p["bbox"] for p in players
            if p.get("cls_id", self.CLS_PLAYER) == self.CLS_PLAYER
        ]
        if not field_bboxes:
            return []
        return list(self.extract_embeddings_batch(frame, field_bboxes))

    def fit(self, all_embeddings: np.ndarray):
        """
        Fit KMeans on player appearance embeddings to discover 2 teams.
        """
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=42,
        )
        self.kmeans.fit(all_embeddings)
        self._fitted = True

        # Assign team IDs by cluster size (larger cluster = Team A)
        labels = self.kmeans.labels_
        counter = Counter(labels)
        sorted_clusters = counter.most_common()

        self.cluster_to_team = {}
        for team_id, (cluster_id, _) in enumerate(sorted_clusters):
            self.cluster_to_team[cluster_id] = team_id

        print(f"Team classification fitted on {len(all_embeddings)} field player embeddings")
        print(f"  Cluster distribution: {dict(counter)}")

    def predict(self, embedding: np.ndarray) -> int:
        """Predict team ID for a single embedding vector."""
        if not self._fitted:
            raise RuntimeError("TeamClassifier not fitted yet. Call fit() first.")
        cluster = self.kmeans.predict(embedding.reshape(1, -1))[0]
        return self.cluster_to_team[cluster]

    # ------------------------------------------------------------------
    # Per-frame classification
    # ------------------------------------------------------------------

    def classify_players(self, frame: np.ndarray, players: list) -> list:
        """
        Add 'team_id' to each player detection dict.

        - Referees (cls_id=3)   → TEAM_REFEREE
        - Others (players / GKs) → batched ResNet embedding → KMeans team
        """
        # Assign referees immediately
        non_refs = []
        for p in players:
            if p.get("cls_id", self.CLS_PLAYER) == self.CLS_REFEREE:
                p["team_id"] = self.TEAM_REFEREE
            else:
                non_refs.append(p)

        # Batch-embed remaining players + goalkeepers
        if non_refs:
            bboxes = [p["bbox"] for p in non_refs]
            embeddings = self.extract_embeddings_batch(frame, bboxes)
            for p, emb in zip(non_refs, embeddings):
                p["team_id"] = self.predict(emb)

        return players
