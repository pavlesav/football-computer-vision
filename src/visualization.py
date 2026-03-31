import cv2
import numpy as np

from .config import Config


class Annotator:
    """Draw annotations on video frames."""

    def __init__(self):
        self.team_colors = Config.TEAM_COLORS
        self.ball_color = Config.BALL_COLOR

    def draw_player_ellipse(self, frame: np.ndarray, bbox: list, team_id: int, track_id: int = -1) -> np.ndarray:
        """Draw a colored ellipse at the player's feet with track ID label."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        bottom_y = y2
        width = (x2 - x1) // 2

        color = self.team_colors.get(team_id, (255, 255, 255))

        # Ellipse at feet
        cv2.ellipse(
            frame,
            center=(center_x, bottom_y),
            axes=(width, int(width * 0.35)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=Config.ELLIPSE_THICKNESS,
        )

        # Track ID label
        if track_id >= 0:
            label = str(track_id)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.FONT_THICKNESS)
            label_x = center_x - tw // 2
            label_y = bottom_y + th + 8

            # Background rectangle for readability
            cv2.rectangle(frame, (label_x - 2, label_y - th - 2), (label_x + tw + 2, label_y + 2), color, -1)
            cv2.putText(
                frame, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, (0, 0, 0),
                Config.FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame

    def draw_ball_triangle(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Draw a small triangle above the ball."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        top_y = y1

        size = 12
        triangle = np.array([
            [center_x, top_y - size - 5],
            [center_x - size, top_y - 5],
            [center_x + size, top_y - 5],
        ], dtype=np.int32)

        cv2.fillPoly(frame, [triangle], self.ball_color)
        cv2.polylines(frame, [triangle], True, (0, 0, 0), 1)

        return frame

    def draw_bbox(self, frame: np.ndarray, bbox: list, label: str, color: tuple) -> np.ndarray:
        """Draw a standard bounding box with label (for initial debugging)."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, (0, 0, 0),
                Config.FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame

    def annotate_frame_debug(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Simple bounding-box annotation for debugging.
        Shows class label + confidence.
        """
        annotated = frame.copy()

        for p in detections["players"]:
            conf = p["conf"]
            tid = p.get("track_id", -1)
            label = f"player {tid} {conf:.2f}" if tid >= 0 else f"player {conf:.2f}"
            self.draw_bbox(annotated, p["bbox"], label, (0, 255, 0))

        for b in detections["ball"]:
            conf = b["conf"]
            label = f"ball {conf:.2f}"
            self.draw_bbox(annotated, b["bbox"], label, self.ball_color)

        return annotated

    def annotate_frame(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """
        Full annotation with team-colored ellipses and ball triangle.
        Requires 'team_id' in player dicts.
        """
        annotated = frame.copy()

        for p in detections["players"]:
            team_id = p.get("team_id", 0)
            track_id = p.get("track_id", -1)
            self.draw_player_ellipse(annotated, p["bbox"], team_id, track_id)

        for b in detections["ball"]:
            self.draw_ball_triangle(annotated, b["bbox"])

        return annotated
