import cv2
import subprocess
import sys
from pathlib import Path

from .config import Config


def extract_clip(
    source: Path = None,
    output: Path = None,
    start_sec: float = None,
    duration_sec: float = None,
) -> Path:
    """Extract a clip from a video using ffmpeg (fast, no re-encoding)."""
    source = source or Config.FULL_MATCH_VIDEO
    output = output or Config.TEST_CLIP
    start_sec = start_sec if start_sec is not None else Config.GAME_START_SEC
    duration_sec = duration_sec if duration_sec is not None else Config.CLIP_DURATION_SEC

    if output.exists():
        print(f"Clip already exists: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    # Convert seconds to HH:MM:SS
    start_ts = _seconds_to_timestamp(start_sec)
    duration_ts = _seconds_to_timestamp(duration_sec)

    cmd = [
        "ffmpeg",
        "-ss", start_ts,
        "-i", str(source),
        "-t", duration_ts,
        "-c", "copy",       # no re-encoding — instant
        "-avoid_negative_ts", "make_zero",
        str(output),
    ]

    print(f"Extracting clip: {start_ts} + {duration_ts} → {output.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
        raise RuntimeError("ffmpeg clip extraction failed")

    print(f"Clip saved: {output}  ({output.stat().st_size / 1e6:.1f} MB)")
    return output


def open_video(path: Path) -> cv2.VideoCapture:
    """Open a video file and return the VideoCapture object."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Return basic video metadata."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def create_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a VideoWriter for mp4 output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def _seconds_to_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
