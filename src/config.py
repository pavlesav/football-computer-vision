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

    # --- YOLO Detection ---
    YOLO_MODEL = r"c:\Users\pavle\runs\detect\football_finetune_v23\weights\best.pt"
    PLAYER_CONF_THRESHOLD = 0.25
    BALL_CONF_THRESHOLD = 0.15
    # Custom fine-tuned model classes
    PLAYER_CLASS_IDS = [0, 2, 3]        # player, goalkeeper, referee
    BALL_CLASS_ID = 1                   # ball

    # --- Team Classification (ResNet18 embeddings) ---
    N_TEAMS = 2

    # --- Tracking ---
    TRACKER_TYPE = "bytetrack"

    # --- Visualization ---
    TEAM_COLORS = {
        0: (255, 50, 50),       # Team A — blue (BGR)
        1: (0, 220, 255),       # Team B — yellow (BGR)
        2: (180, 180, 180),     # Referee — gray
    }
    BALL_COLOR = (0, 200, 0)    # Green (BGR)
    ELLIPSE_THICKNESS = 2
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1

    # --- Processing ---
    FRAME_SAMPLE_RATE = 1              # process every frame (increase for speed)
    BATCH_SIZE_EMBEDDING = 64           # frames to collect before team classification
