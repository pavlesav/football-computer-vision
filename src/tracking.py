from .config import Config


class Tracker:
    """
    Track management and team assignment.

    Supports two modes:
      - **Track-level** (preferred): classify each track once from its
        mean embedding via :meth:`set_track_teams` / :meth:`get_team`.
      - **Per-frame vote** (legacy fallback): majority vote over
        per-frame classifications via :meth:`get_stable_team`.
    """

    def __init__(self):
        # Track-level assignments (set once after GMM fit)
        self.track_teams = {}

        # Per-frame majority vote (legacy)
        self.track_team_history = {}
        self.track_team_cache = {}

    # ── Track-level classification (preferred) ────────────────────────

    def set_track_teams(self, track_teams: dict):
        """Store track-level team assignments (one classification per track)."""
        self.track_teams = track_teams

    def get_team(self, track_id: int, fallback: int = 0) -> int:
        """Get team for a track. Returns *fallback* if track is unknown."""
        return self.track_teams.get(track_id, fallback)

    # ── Per-frame majority vote (legacy) ──────────────────────────────

    def update_team_assignment(self, track_id: int, team_id: int):
        """Record a team assignment for a tracked player."""
        if track_id not in self.track_team_history:
            self.track_team_history[track_id] = []
        self.track_team_history[track_id].append(team_id)

        history = self.track_team_history[track_id]
        self.track_team_cache[track_id] = max(set(history), key=history.count)

    def get_stable_team(self, track_id: int, current_team_id: int) -> int:
        """Get temporally smoothed team ID via majority vote."""
        self.update_team_assignment(track_id, current_team_id)
        return self.track_team_cache.get(track_id, current_team_id)

    def reset(self):
        """Clear all tracking history."""
        self.track_teams.clear()
        self.track_team_history.clear()
        self.track_team_cache.clear()
