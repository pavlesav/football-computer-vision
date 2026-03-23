from .config import Config


class Tracker:
    """
    Wrapper around YOLO's built-in ByteTrack.

    ByteTrack tracking is handled directly in PlayerDetector.detect_with_tracking().
    This module provides utilities for track management and ID mapping.
    """

    def __init__(self):
        self.track_team_history = {}  # track_id -> list of team_id assignments
        self.track_team_cache = {}    # track_id -> most common team_id

    def update_team_assignment(self, track_id: int, team_id: int):
        """Record a team assignment for a tracked player."""
        if track_id not in self.track_team_history:
            self.track_team_history[track_id] = []
        self.track_team_history[track_id].append(team_id)

        # Update cache with majority vote (temporal smoothing)
        history = self.track_team_history[track_id]
        self.track_team_cache[track_id] = max(set(history), key=history.count)

    def get_stable_team(self, track_id: int, current_team_id: int) -> int:
        """
        Get temporally smoothed team ID for a track.

        Uses majority vote over recent assignments to prevent
        flickering between teams frame-to-frame.
        """
        self.update_team_assignment(track_id, current_team_id)
        return self.track_team_cache.get(track_id, current_team_id)

    def reset(self):
        """Clear all tracking history."""
        self.track_team_history.clear()
        self.track_team_cache.clear()
