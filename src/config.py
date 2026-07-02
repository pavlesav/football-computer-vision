from pathlib import Path


class Config:
    # --- Paths ---
    PROJECT_ROOT          = Path(__file__).resolve().parent.parent
    VIDEOS_DIR            = PROJECT_ROOT / "videos"
    OUTPUT_DIR             = PROJECT_ROOT / "output"
    OUTPUT_TEAMS_DIR       = OUTPUT_DIR / "teams"
    OUTPUT_HOMOGRAPHY_DIR  = OUTPUT_DIR / "homography"
    OUTPUT_EVENTS_DIR      = OUTPUT_DIR / "events"
    OUTPUT_GAME_STATE_DIR  = OUTPUT_DIR / "game_state"
    OUTPUT_CLASSIFIERS_DIR = OUTPUT_DIR / "classifiers"

    # --- Match videos (all 16 annotated matches) ---
    MATCH_VIDEOS = {
        "ars-dec":   VIDEOS_DIR / "ARSENAL - DECIC 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
        "bok-jed":   VIDEOS_DIR / "BOKELJ - JEDINSTVO 1.CFL 22.KOLO 01.03.2026. CIJELA UTAKMICA.mp4",
        "bud-sut":   VIDEOS_DIR / "BUDUCNOST - SUTJESKA 1.CFL 20.KOLO 21.02.2026. CIJELA UTAKMICA.mp4",
        "jez-jed":   VIDEOS_DIR / "JEZERO - JEDINSTVO 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
        "pet-mor":   VIDEOS_DIR / "PETROVAC - MORNAR 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
        "sut-mla":   VIDEOS_DIR / "SUTJESKA - MLADOST 1.CFL 21.03.2026. CIJELA UTAKMICA.mp4",
        "bok-jed-2": VIDEOS_DIR / "BOKELJ-JEDINSTVO 1.CFL 4.KOLO 24.08.2025 CIJELA.mp4",
        "dec-mla":   VIDEOS_DIR / "DECIC-MLADOST DG 1.CFL 1.KOLO 04.08.2025 CIJELA.mp4",
        "jed-ars":   VIDEOS_DIR / "JEDINSTVO-ARSENAL 1.CFL 10.KOLO 01.10.2025 CIJELA.mp4",
        "jez-ars":   VIDEOS_DIR / "JEZERO-ARSENAL 1.CFL 14.KOLO 02.11.2025 CIJELA.mp4",
        "mla-bud":   VIDEOS_DIR / "MLADOST DG - BUDUCNOST 1.CFL 23. KOLO 07.03.2026. CIJELA UTAKMICA.mp4",
        "mla-bud-2": VIDEOS_DIR / "MLADOST DG-BUDUCNOST 1.CFL 5.KOLO 31.08.2025. CIJELA.mp4",
        "mor-ars":   VIDEOS_DIR / "MORNAR-ARSENAL 1.CFL 2.KOLO 10.08.2025. CIJELA.mp4",
        "mor-bud":   VIDEOS_DIR / "MORNAR-BUDUCNOST 1.CFL 7.KOLO 17.09.2025 CIJELA.mp4",
        "pet-bok":   VIDEOS_DIR / "PETROVAC-BOKELJ 1.CFL 19.KOLO 10.12.2025 CIJELA.mp4",
        "sut-pet":   VIDEOS_DIR / "SUTJESKA-PETROVAC 1.CFL 3.KOLO 17.08.2025 CIJELA.mp4",
    }

    # --- YOLO Detection ---
    YOLO_MODEL = None
    YOLO_SEG_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "segmentation" / "yolov8n-seg.pt")
    PLAYER_CONF_THRESHOLD = 0.25
    BALL_CONF_THRESHOLD = 0.15
    # 2-class model: YOLO detects person (0) and ball (1) only.
    # Goalkeeper / referee / team assignment is handled in post-processing.
    PLAYER_CLASS_IDS = [0]              # person
    BALL_CLASS_ID = 1                   # ball

    # --- Team Classification (ResNet18 embeddings) ---
    N_TEAMS = 2

    # --- Broadcast Analysis ---
    # Scoreboard ROI (x1, y1, x2, y2) for 1920x1080 1.CFL broadcasts.
    # Covers the top-left scoreboard bar (clock, team names, score).
    SCOREBOARD_ROI = (40, 44, 520, 88)
    # Wide search region (x1, y1, x2, y2) where the match clock may appear.
    # Covers all three 1.CFL scoreboard layouts we've seen:
    #   - Layout A: clock on the left, y=44-88  (bud-sut, pet-mor)
    #   - Layout B: clock on the right, y=44-88 (ars-dec, jed-ars, mla-bud)
    #   - Layout C: scoreboard ~40px lower      (jez-jed)
    # easyocr runs over this whole region and finds any MM:SS-like text.
    CLOCK_SEARCH_ROI = (30, 30, 650, 140)

    # --- Tracking ---
    TRACKER_TYPE = "botsort"

    # --- Visualization ---
    TEAM_COLORS = {
        0: (255, 50, 50),       # Team A — blue (BGR)
        1: (0, 220, 255),       # Team B — yellow (BGR)
        2: (180, 180, 180),     # Other — gray (ref/GK/staff)
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
          2) models/detection/weights/best.pt (fine-tuned model)
          3) Fallback to base pretrained model
        """
        if cls.YOLO_MODEL:
            configured = Path(cls.YOLO_MODEL)
            if configured.exists():
                return str(configured)

        detection_weights = cls.PROJECT_ROOT / "models" / "detection" / "weights" / "best.pt"
        if detection_weights.exists():
            return str(detection_weights)

        return "yolov8n.pt"
