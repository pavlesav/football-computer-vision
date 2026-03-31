from pathlib import Path


class Config:
    # --- Paths ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    VIDEOS_DIR = PROJECT_ROOT / "videos"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    FULL_MATCH_VIDEO = VIDEOS_DIR / "BUDUCNOST - SUTJESKA 1.CFL 20.KOLO 21.02.2026. CIJELA UTAKMICA.mp4"
    TEST_CLIP = VIDEOS_DIR / "test_clip_5min.mp4"

    # --- Clip extraction ---
    GAME_START_SEC = 4 * 60 + 38        # 04:38
    CLIP_DURATION_SEC = 5 * 60           # 5 minutes

    # --- Ball annotation video sources ---
    # Add new match videos to videos/ and register them here.
    # Each entry:
    #   path      – Path to the video file in videos/
    #   slug      – Short unique prefix used in extracted frame filenames (no spaces)
    #   n_frames  – Frames to sample, spread evenly across the full match
    #   seed      – Unique seed per match → independent random samples
    BALL_ANNOTATION_VIDEOS = [
        # bud-sut is excluded — already have 300 fully-labeled frames from that match
        {
            "path": VIDEOS_DIR / "ARSENAL - DECIC 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
            "slug": "ars-dec",
            "n_frames": 60,
            "seed": 43,
        },
        {
            "path": VIDEOS_DIR / "BOKELJ - JEDINSTVO 1.CFL 22.KOLO 01.03.2026. CIJELA UTAKMICA.mp4",
            "slug": "bok-jed",
            "n_frames": 60,
            "seed": 44,
        },
        {
            "path": VIDEOS_DIR / "JEZERO - JEDINSTVO 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
            "slug": "jez-jed",
            "n_frames": 60,
            "seed": 45,
        },
        {
            "path": VIDEOS_DIR / "PETROVAC - MORNAR 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
            "slug": "pet-mor",
            "n_frames": 60,
            "seed": 46,
        },
        {
            "path": VIDEOS_DIR / "SUTJESKA - MLADOST 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
            "slug": "sut-mla",
            "n_frames": 60,
            "seed": 47,
        },
    ]

    # --- YOLO Detection ---
    YOLO_MODEL = None
    YOLO_SEG_MODEL = "yolov8n-seg.pt"  # segmentation model for team classification
    PLAYER_CONF_THRESHOLD = 0.25
    BALL_CONF_THRESHOLD = 0.15
    # 2-class model: YOLO detects person (0) and ball (1) only.
    # Goalkeeper / referee / team assignment is handled in post-processing.
    PLAYER_CLASS_IDS = [0]              # person
    BALL_CLASS_ID = 1                   # ball

    # --- Team Classification (ResNet18 embeddings) ---
    N_TEAMS = 2

    # --- Tracking ---
    TRACKER_TYPE = "botsort"

    # --- Visualization ---
    TEAM_COLORS = {
        0: (255, 50, 50),       # Team A — blue (BGR)
        1: (0, 220, 255),       # Team B — yellow (BGR)
    }
    BALL_COLOR = (0, 200, 0)    # Green (BGR)
    ELLIPSE_THICKNESS = 2
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1

    # --- Processing ---
    FRAME_SAMPLE_RATE = 1              # process every frame (increase for speed)
    BATCH_SIZE_EMBEDDING = 64           # frames to collect before team classification

    @classmethod
    def resolve_yolo_model(cls) -> str:
        """
        Resolve YOLO weights path for inference.

        Priority:
          1) Explicit Config.YOLO_MODEL if set and exists
          2) Most recently modified local runs/detect/*/weights/best.pt
          3) Fallback to base pretrained model
        """
        if cls.YOLO_MODEL:
            configured = Path(cls.YOLO_MODEL)
            if configured.exists():
                return str(configured)

        runs_dir = cls.PROJECT_ROOT / "runs" / "detect"
        if runs_dir.exists():
            checkpoints = sorted(
                runs_dir.glob("*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if checkpoints:
                return str(checkpoints[0])

        return "yolov8n.pt"
