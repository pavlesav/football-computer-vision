from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from .config import Config


class BallInterpolator:
    """Fill short ball detection gaps with linear bbox interpolation."""

    def __init__(self, max_gap: int = 15):
        self.max_gap = max_gap

    def interpolate(self, frame_detections: List[dict]) -> List[dict]:
        """
        Fill frames where ball is missing by linearly interpolating the bbox
        between the nearest detected positions before and after the gap.

        Args:
            frame_detections: list of per-frame detection dicts, each with a
                              'ball' key (list of ball detections).

        Returns:
            The same list, modified in-place with interpolated ball entries.
        """
        # Collect frames where ball was detected: (frame_idx, bbox)
        ball_frames = []
        for i, det in enumerate(frame_detections):
            if det["ball"]:
                ball_frames.append((i, det["ball"][0]["bbox"]))

        # Interpolate gaps between consecutive detections
        for j in range(len(ball_frames) - 1):
            idx_a, bbox_a = ball_frames[j]
            idx_b, bbox_b = ball_frames[j + 1]
            gap = idx_b - idx_a - 1

            if 0 < gap <= self.max_gap:
                for k in range(1, gap + 1):
                    t = k / (gap + 1)
                    interp_bbox = [
                        int(bbox_a[c] + t * (bbox_b[c] - bbox_a[c]))
                        for c in range(4)
                    ]
                    frame_detections[idx_a + k]["ball"] = [
                        {
                            "bbox": interp_bbox,
                            "conf": 0.0,
                            "cls_id": Config.BALL_CLASS_ID,
                            "interpolated": True,
                        }
                    ]

        return frame_detections


class PlayerDetector:
    """YOLO-based detector for players and ball."""

    def __init__(self, model_path: str = None):
        model_path = model_path or Config.resolve_yolo_model()
        self.player_conf = Config.PLAYER_CONF_THRESHOLD
        self.ball_conf = Config.BALL_CONF_THRESHOLD
        # Run YOLO at the lower threshold, filter per-class after
        self.conf = min(self.player_conf, self.ball_conf)
        self.device = self._resolve_device()
        self.model = YOLO(model_path)
        self.player_classes = set(Config.PLAYER_CLASS_IDS)
        self.ball_class = Config.BALL_CLASS_ID

    def _resolve_device(self):
        """Use CUDA only when the active torch build supports the detected GPU arch."""
        if not torch.cuda.is_available():
            return "cpu"

        try:
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{cc_major}{cc_minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            if device_arch in supported_arches:
                return 0
        except Exception:
            pass

        return "cpu"

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
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
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
            device=self.device,
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
