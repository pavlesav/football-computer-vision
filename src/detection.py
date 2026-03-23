import numpy as np
from ultralytics import YOLO

from .config import Config


class PlayerDetector:
    """YOLO-based detector for players and ball."""

    def __init__(self, model_path: str = None):
        model_path = model_path or Config.YOLO_MODEL
        self.player_conf = Config.PLAYER_CONF_THRESHOLD
        self.ball_conf = Config.BALL_CONF_THRESHOLD
        # Run YOLO at the lower threshold, filter per-class after
        self.conf = min(self.player_conf, self.ball_conf)
        self.model = YOLO(model_path)
        self.player_classes = set(Config.PLAYER_CLASS_IDS)
        self.ball_class = Config.BALL_CLASS_ID

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single frame.

        Returns:
            dict with keys:
                - 'players': list of dicts {bbox, conf}
                - 'ball': list of dicts {bbox, conf}
                - 'raw_results': ultralytics Results object
            bbox format: [x1, y1, x2, y2] (pixel coords)
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes = results.boxes

        players = []
        ball = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i])

            det = {"bbox": bbox, "conf": conf, "cls_id": cls_id}

            if cls_id in self.player_classes and conf >= self.player_conf:
                players.append(det)
            elif cls_id == self.ball_class and conf >= self.ball_conf:
                ball.append(det)

        return {
            "players": players,
            "ball": ball,
            "raw_results": results,
        }

    def detect_with_tracking(self, frame: np.ndarray) -> dict:
        """
        Run detection + ByteTrack tracking on a single frame.

        Returns:
            dict with keys:
                - 'players': list of dicts {bbox, conf, track_id}
                - 'ball': list of dicts {bbox, conf, track_id}
                - 'raw_results': ultralytics Results object
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            tracker=f"{Config.TRACKER_TYPE}.yaml",
            persist=True,
            verbose=False,
        )[0]
        boxes = results.boxes

        players = []
        ball = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i])
            track_id = int(boxes.id[i]) if boxes.id is not None else -1

            det = {"bbox": bbox, "conf": conf, "track_id": track_id, "cls_id": cls_id}

            if cls_id in self.player_classes and conf >= self.player_conf:
                players.append(det)
            elif cls_id == self.ball_class and conf >= self.ball_conf:
                ball.append(det)

        return {
            "players": players,
            "ball": ball,
            "raw_results": results,
        }
