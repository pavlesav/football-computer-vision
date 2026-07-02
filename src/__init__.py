from .config import Config


def __getattr__(name):
    _lazy = {
        "extract_clip": ".video_utils",
        "open_video": ".video_utils",
        "get_video_info": ".video_utils",
        "create_video_writer": ".video_utils",
        "PlayerDetector": ".detection",
        "BallInterpolator": ".detection",
        "TeamClassifier": ".team_classifier",
        "Tracker": ".tracking",
        "Annotator": ".visualization",
        "analyze_broadcast": ".broadcast",
        "detect_cuts": ".broadcast",
        "build_segments": ".broadcast",
        "check_scoreboard": ".broadcast",
        "find_game_periods": ".broadcast",
        "detect_period_starts": ".broadcast",
        "read_clock_seconds": ".broadcast",
        "validate_period_starts": ".broadcast",
        "estimate_homography": ".homography",
        "GameState": ".game_state",
        "track_ball": ".ball_tracker",
        "BallTrackerParams": ".ball_tracker",
        "detect_events": ".events",
        "export_events": ".events",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
