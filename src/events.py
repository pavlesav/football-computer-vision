"""
Rule-based event detection on the persisted :mod:`src.game_state`.

This is the analysis half of the pipeline — the actual deliverable. It reads the
game-state artifact (per-frame players with pitch XY + team, ball pitch XY) and
derives a StatsBomb-lite event stream: possession, passes, carries, shots, and
possession changes.

Approach (the "possession-then-event" decision-tree method from the tracking-data
literature — Anzer/Bauer, PLOS One 2024): first decide who controls the ball each
frame, collapse that into discrete *touches*, then classify each touch→touch
transition. Rule-based possession→events reaches F≈0.9 on clean tracking data;
our accuracy is bounded by ball/homography quality, so events are **only emitted
on frames with a valid camera projection** (confidence gating) — loose frames
simply carry no possession.

Coordinates: native is metres on 105×68 (pitch origin at a corner). The export
also provides StatsBomb 120×80 ``location`` fields.

Usage::

    python -m src.events --match sut-mla
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .game_state import (GameState, trusted_frame_mask, adaptive_conf_min)
from .ball_tracker import track_ball
from .roles import infer_attack_direction, identify_goalkeepers

# ── Tunables (metres / frames) ───────────────────────────────────────────────
# Homography trust gate: see game_state.trusted_frame_mask — wide shot AND
# line-alignment confidence above a per-match adaptive threshold (fixed 0.75
# did not transfer across matches; the score scale is match-dependent).
# Player/ball pitch positions on untrusted frames are dropped so a bad
# projection can't poison possession. Pass conf_min explicitly to override.
POSSESSION_RADIUS_M = 3.0     # ball within this of a player's feet ⇒ controlled
CARRIER_MAX_BALL_SPEED = 9.0  # m/s — faster than this is flying past, not controlled
CARRIER_MAX_MISSED = 10       # coasted ball older than this can't assign possession
MIN_TOUCH_FRAMES = 3          # a carrier must hold ≥ this to count as a touch
CARRIER_GAP_FRAMES = 12       # merge same-player touches split by a shorter gap
# A team only *gains possession* if its spell lasts >= this many frames or
# contains >= 2 touches. A lone sub-second touch between opponent touches is a
# deflection/duel: QC showed those emitted 3 "Possession Change" events in
# 0.9s during one midfield scramble. StatsBomb possession sequences behave the
# same way (a blocked pass does not start an opponent possession).
MIN_SPELL_FRAMES = 25         # 1.0s @ 25fps
SPELL_MIN_TOUCHES = 2
# Dead-ball handling: once the ball crosses the pitch boundary it is out of
# play until it re-enters and settles; carrier assignment during retrieval /
# throw-in setup minted false passes (golden segA: retriever->thrower logged
# as a completed pass).
OUT_MARGIN_M = 0.5            # beyond the line by this ⇒ out of play
DEAD_BALL_COOLDOWN = 12       # frames back in bounds before play resumes
# Kick detection: a sharp ball-velocity discontinuity next to a player is a
# touch even if the player was never slow-ball-within-3m (golden segA FN: a
# defender ran onto a rolling ball and cleared it first-time at full stride —
# proximity+speed gating alone cannot see running one-touches).
KICK_DV_MS = 6.0              # velocity change (m/s) that counts as a kick
KICK_RADIUS_M = 4.0           # kicker must be within this of the ball
MIN_PASS_DIST_M = 4.0         # travel below this on a turnover ⇒ duel, not a pass
MIN_CARRY_DIST_M = 3.0        # displacement within a touch to log a Carry
SHOT_FINAL_THIRD_M = 70.0     # touch beyond this (or before 35) can start a shot
SHOT_BALL_SPEED_MS = 9.0      # ball must move at least this fast toward goal
GOAL_Y_MIN_M, GOAL_Y_MAX_M = 30.34, 37.66   # goalmouth width (+ small margin)

PITCH_L, PITCH_W = 105.0, 68.0


# ── Touch / event containers ─────────────────────────────────────────────────

@dataclass
class Touch:
    track_id: int
    team: int
    f_start: int
    f_end: int
    xy_start: tuple
    xy_end: tuple


@dataclass
class Event:
    index: int
    period: int
    frame: int
    time_sec: float
    type: str                       # Pass | Carry | Shot | Possession Change
    team: int
    player: int
    location: tuple                 # metres (x, y)
    details: dict = field(default_factory=dict)


# ── Ball trajectory ──────────────────────────────────────────────────────────

def ball_series(gs: GameState, conf_min: Optional[float] = None) -> pd.DataFrame:
    """Frame-indexed ball pitch position + speed from the Kalman tracker
    (:mod:`src.ball_tracker`). The tracker only ingests detections on trusted
    frames (same gate as above) and coasts physically through detection gaps —
    replacing the old drop-then-interpolate approach whose ≤8-frame limit could
    not bridge the 12-64 frame gaps YOLO actually loses the ball for."""
    b = track_ball(gs, conf_min=conf_min)
    b = b.rename(columns={"x": "bx", "y": "by"})
    return b[["frame", "time_sec", "bx", "by", "vx", "vy", "speed",
              "source", "missed"]]


# ── Possession / carrier ─────────────────────────────────────────────────────

def dead_ball_frames(ball: pd.DataFrame) -> set:
    """Frames where the ball is out of play: from the moment it crosses the
    pitch boundary until it has been back in bounds for DEAD_BALL_COOLDOWN
    (the restart). Shared by carrier assignment AND kick detection — kicks
    minted during throw-in retrieval fabricated passes (golden segA)."""
    dead = set()
    dead_state = False
    back_in_since = None
    for r in ball.itertuples(index=False):
        if not (r.bx == r.bx):
            continue
        f = int(r.frame)
        oob = (r.bx < -OUT_MARGIN_M or r.bx > PITCH_L + OUT_MARGIN_M
               or r.by < -OUT_MARGIN_M or r.by > PITCH_W + OUT_MARGIN_M)
        if oob:
            dead_state = True
            back_in_since = None
        elif dead_state:
            if back_in_since is None:
                back_in_since = f
            elif f - back_in_since >= DEAD_BALL_COOLDOWN:
                dead_state = False
        if dead_state:
            dead.add(f)
    return dead


def _effective_players(gs: GameState, trusted: set,
                       team_override: Optional[dict]) -> pd.DataFrame:
    """Players eligible for possession on trusted frames. ``team_override``
    (track_id → team) folds goalkeepers in: GKs are classified 'Other' by the
    kit classifier and were invisible to possession — the golden set measured
    the GK build-up pass as a false negative."""
    pl = gs.players
    pl = pl[pl.pitch_x.notna() & pl.frame.isin(trusted)]
    if team_override:
        pl = pl.copy()
        eff = pl["track_id"].map(team_override)
        pl["team_id"] = eff.fillna(pl["team_id"]).astype(int)
    return pl[pl.team_id.isin([0, 1])]


def carrier_per_frame(gs: GameState, ball: pd.DataFrame,
                      conf_min: Optional[float] = None,
                      dead: Optional[set] = None,
                      team_override: Optional[dict] = None) -> dict:
    """{frame: (track_id, team)} for trusted frames where a player controls the ball."""
    if dead is None:
        dead = dead_ball_frames(ball)
    trusted = set(gs.frames.loc[trusted_frame_mask(gs, conf_min), "frame"].astype(int))
    pl = _effective_players(gs, trusted, team_override)
    by_frame: dict[int, list] = defaultdict(list)
    for r in pl.itertuples(index=False):
        by_frame[r.frame].append((r.track_id, r.team_id, r.pitch_x, r.pitch_y))

    out = {}
    for r in ball.itertuples(index=False):
        if not (r.bx == r.bx):        # NaN ball
            continue
        f = int(r.frame)
        if f in dead or f not in trusted:
            continue
        # A ball estimate coasting unseen for too long must not keep extending
        # a touch — that staleness is what merged passes before. Bridged frames
        # are exempt: both endpoints of a bridge are real detections.
        if r.source == "kalman" and r.missed == r.missed \
                and r.missed > CARRIER_MAX_MISSED:
            continue
        # A ball moving faster than anyone can control is in transit — a
        # player it happens to sweep past is not its carrier. Full-half QC
        # showed these fly-by assignments minted false touches (and ~15% extra
        # passes); receptions still register because controlling the ball
        # slows it below the gate within a frame or two.
        if r.speed == r.speed and r.speed > CARRIER_MAX_BALL_SPEED:
            continue
        cands = by_frame.get(r.frame)
        if not cands:
            continue
        best_d, best = 1e9, None
        for tid, team, px, py in cands:
            d = (px - r.bx) ** 2 + (py - r.by) ** 2
            if d < best_d:
                best_d, best = d, (tid, team)
        if best is not None and best_d <= POSSESSION_RADIUS_M ** 2:
            out[int(r.frame)] = best
    return out


def detect_kicks(gs: GameState, ball: pd.DataFrame,
                 conf_min: Optional[float] = None,
                 dead: Optional[set] = None,
                 team_override: Optional[dict] = None) -> list[tuple]:
    """[(frame, tid, team)] where the ball's velocity jumps by >= KICK_DV_MS
    with a team-0/1 player within KICK_RADIUS_M — evidence someone kicked it.
    Only frames where the jump lands on a detected/bridged estimate qualify
    (a coast can't produce a real discontinuity)."""
    if dead is None:
        dead = dead_ball_frames(ball)
    trusted = set(gs.frames.loc[trusted_frame_mask(gs, conf_min),
                                "frame"].astype(int))
    trusted -= dead
    pl = _effective_players(gs, trusted, team_override)
    by_frame: dict[int, list] = defaultdict(list)
    for r in pl.itertuples(index=False):
        by_frame[r.frame].append((r.track_id, r.team_id, r.pitch_x, r.pitch_y))

    b = ball.dropna(subset=["bx"]).reset_index(drop=True)
    kicks = []
    for i in range(1, len(b)):
        r0, r1 = b.iloc[i - 1], b.iloc[i]
        if int(r1.frame) - int(r0.frame) > 3:
            continue
        if r1.source not in ("detected", "bridged"):
            continue
        dv = float(np.hypot(r1.vx - r0.vx, r1.vy - r0.vy))
        if dv < KICK_DV_MS:
            continue
        f = int(r1.frame)
        if f not in trusted:
            continue
        cands = by_frame.get(f) or by_frame.get(f - 1)
        if not cands:
            continue
        best_d, best = 1e9, None
        for tid, team, px, py in cands:
            d = float(np.hypot(px - r0.bx, py - r0.by))
            if d < best_d:
                best_d, best = d, (tid, team)
        if best is not None and best_d <= KICK_RADIUS_M:
            kicks.append((f, best[0], best[1]))
    return kicks


def build_touches(gs: GameState, carrier: dict,
                  ball: Optional[pd.DataFrame] = None,
                  kicks: Optional[list] = None) -> list[Touch]:
    """Collapse the per-frame carrier map into debounced touches.

    If ``ball`` is given, a touch must overlap at least one frame where the
    ball was actually *detected* — touches built purely on bridged/coasted
    estimates are fabrications (golden segB: micro-touches minted inside an
    aerial blackout while the real ball was high in the air)."""
    detected_frames = None
    if ball is not None:
        detected_frames = set(
            ball.loc[ball["source"] == "detected", "frame"].astype(int))
    frames = gs.frames.sort_values("frame")["frame"].tolist()
    # Position lookup: (frame, tid) -> (x, y)
    pos = {(int(r.frame), int(r.track_id)): (r.pitch_x, r.pitch_y)
           for r in gs.players.itertuples(index=False) if r.pitch_x == r.pitch_x}

    # Raw runs of constant carrier tid (None where no carrier).
    runs = []   # (tid, team, f_start, f_end)
    cur = None
    for f in frames:
        c = carrier.get(int(f))
        tid = c[0] if c else None
        if cur and cur[0] == tid:
            cur[3] = f
        else:
            if cur:
                runs.append(tuple(cur))
            cur = [tid, (c[1] if c else None), f, f] if tid is not None else None
            if tid is None:
                cur = None
    if cur:
        runs.append(tuple(cur))

    # Merge same-tid runs separated by a short gap (brief ball loss on a dribble).
    merged = []
    for run in runs:
        if merged and merged[-1][0] == run[0] and \
                (run[2] - merged[-1][3]) <= CARRIER_GAP_FRAMES:
            merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2], run[3])
        else:
            merged.append(run)

    touches = []
    for tid, team, f0, f1 in merged:
        if (f1 - f0 + 1) < MIN_TOUCH_FRAMES:
            continue
        if detected_frames is not None and not any(
                f in detected_frames for f in range(int(f0), int(f1) + 1)):
            continue
        xy0 = pos.get((f0, tid))
        xy1 = pos.get((f1, tid))
        if xy0 is None or xy1 is None:
            continue
        touches.append(Touch(int(tid), int(team), int(f0), int(f1), xy0, xy1))

    # Inject kick-evidence touches (running one-touches the proximity logic
    # cannot see). Skip kicks already covered by a proximity touch of the
    # same player nearby in time.
    if kicks:
        pos_any = {}
        for (fr, tid), xy in pos.items():
            pos_any.setdefault((fr, tid), xy)
        for f, tid, team in kicks:
            covered = any(t.track_id == tid
                          and t.f_start - CARRIER_GAP_FRAMES <= f
                          <= t.f_end + CARRIER_GAP_FRAMES
                          for t in touches)
            if covered:
                continue
            xy = None
            for ff in (f, f - 1, f + 1, f - 2, f + 2):
                xy = pos_any.get((ff, tid))
                if xy is not None:
                    break
            if xy is None:
                continue
            touches.append(Touch(int(tid), int(team), int(f), int(f), xy, xy))
        touches.sort(key=lambda t: t.f_start)
    return touches


# ── Event derivation ─────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


SHOT_MAX_RANGE_M = 40.0       # touches further than this from the target goal


def _shot_after(touch: Touch, ball: pd.DataFrame,
                goal_x: Optional[float] = None) -> Optional[tuple]:
    """If the ball rockets toward the attacked goal mouth right after this
    touch, return (end_xy, outcome); else None.

    ``goal_x`` is the goal the touching team ATTACKS (from
    roles.infer_attack_direction). The old fallback guessed "whichever end is
    nearer", which turned defensive clearances near a team's own goal into
    'shots' at it."""
    seg = ball[(ball.frame >= touch.f_end) & (ball.frame <= touch.f_end + 40)]
    seg = seg.dropna(subset=["bx", "by"])
    if len(seg) < 3:
        return None
    start = touch.xy_end
    if goal_x is None:
        goal_x = PITCH_L if start[0] >= PITCH_L / 2 else 0.0
    if abs(goal_x - start[0]) > SHOT_MAX_RANGE_M:
        return None
    end = (seg.iloc[-1]["bx"], seg.iloc[-1]["by"])
    max_speed = float(seg["speed"].max())
    toward = abs(end[0] - goal_x) < abs(start[0] - goal_x)     # ball moved goalward
    near_goal = abs(end[0] - goal_x) < 12.0
    in_mouth = GOAL_Y_MIN_M - 4 <= end[1] <= GOAL_Y_MAX_M + 4
    if toward and near_goal and in_mouth and max_speed >= SHOT_BALL_SPEED_MS:
        return end, "attempt"
    return None


def possession_spells(touches: list[Touch]) -> list[dict]:
    """Group consecutive same-team touches into spells and mark which are
    *real* possession (>= MIN_SPELL_FRAMES or >= SPELL_MIN_TOUCHES touches).
    A lone sub-second opponent touch between a team's touches is a
    deflection/duel, not a possession change."""
    spells = []
    for i, t in enumerate(touches):
        if spells and spells[-1]["team"] == t.team:
            spells[-1]["touches"].append(i)
            spells[-1]["f_end"] = t.f_end
        else:
            spells.append({"team": t.team, "touches": [i],
                           "f_start": t.f_start, "f_end": t.f_end})
    for s in spells:
        s["real"] = (len(s["touches"]) >= SPELL_MIN_TOUCHES
                     or (s["f_end"] - s["f_start"] + 1) >= MIN_SPELL_FRAMES)
    return spells


def detect_events(gs: GameState,
                  conf_min: Optional[float] = None) -> tuple[list[Event], dict]:
    if conf_min is None:
        conf_min = adaptive_conf_min(gs)
    ball = ball_series(gs, conf_min)
    dead = dead_ball_frames(ball)
    directions = infer_attack_direction(gs, conf_min)
    gk_map = identify_goalkeepers(gs, directions)
    carrier = carrier_per_frame(gs, ball, conf_min, dead, gk_map)
    kicks = detect_kicks(gs, ball, conf_min, dead, gk_map)
    touches = build_touches(gs, carrier, ball, kicks)
    spells = possession_spells(touches)

    fps = gs.fps
    period = int(gs.meta.get("period", 1))
    events: list[Event] = []

    def add(frame, etype, team, player, loc, details):
        events.append(Event(len(events), period, int(frame), _tsec(gs, frame),
                             etype, int(team), int(player), tuple(loc), details))

    # spell index for each touch + the team holding *real* possession before
    # each spell (for turnover semantics).
    spell_of = {}
    for si, s in enumerate(spells):
        for ti in s["touches"]:
            spell_of[ti] = si

    possession_team = None            # team of the last real spell seen
    for si, s in enumerate(spells):
        if s["real"] and s["team"] != possession_team:
            if possession_team is not None:
                ft = touches[s["touches"][0]]
                add(ft.f_start, "Possession Change", s["team"], ft.track_id,
                    ft.xy_start, {"from_team": int(possession_team)})
            possession_team = s["team"]

    for i, t in enumerate(touches):
        # Carry within the touch.
        if _dist(t.xy_start, t.xy_end) >= MIN_CARRY_DIST_M:
            add(t.f_start, "Carry", t.team, t.track_id, t.xy_start,
                {"end_location": list(t.xy_end),
                 "length": round(_dist(t.xy_start, t.xy_end), 2)})

        # Shot off this touch (toward the goal this team actually attacks).
        shot = _shot_after(t, ball, directions.goal_x(t.team))
        if shot is not None:
            end_xy, outcome = shot
            add(t.f_end, "Shot", t.team, t.track_id, t.xy_end,
                {"end_location": list(end_xy), "outcome": outcome})

        # Transition to the next touch → pass / turnover.
        if i + 1 >= len(touches):
            continue
        nxt = touches[i + 1]
        travel = _dist(t.xy_end, nxt.xy_start)
        if t.team == nxt.team and nxt.track_id != t.track_id:
            add(t.f_end, "Pass", t.team, t.track_id, t.xy_end,
                {"end_location": list(nxt.xy_start), "recipient": int(nxt.track_id),
                 "outcome": "complete", "length": round(travel, 2)})
        elif t.team != nxt.team:
            # Ball reached an opponent. If their spell is real possession it
            # was won (interception); if it is a mere deflection the pass just
            # failed (and possession never changes hands).
            if travel >= MIN_PASS_DIST_M:
                gained = spells[spell_of[i + 1]]["real"]
                add(t.f_end, "Pass", t.team, t.track_id, t.xy_end,
                    {"end_location": list(nxt.xy_start),
                     "outcome": "interception" if gained else "incomplete",
                     "length": round(travel, 2)})

    events.sort(key=lambda e: (e.frame, e.index))
    for k, e in enumerate(events):
        e.index = k
    summary = _summary(events, carrier, gs, conf_min)
    summary["attack_ltr"] = {int(k): bool(v)
                             for k, v in directions.attack_ltr.items()}
    summary["attack_direction_confidence"] = round(directions.confidence, 3)
    summary["gk_tracks"] = {int(k): int(v) for k, v in gk_map.items()}
    return events, summary


def _tsec(gs: GameState, frame: int) -> float:
    row = gs.frames.loc[gs.frames.frame == frame]
    return float(row.iloc[0]["time_sec"]) if not row.empty else frame / gs.fps


def _summary(events, carrier, gs, conf_min) -> dict:
    poss = defaultdict(int)
    for _, team in carrier.values():
        poss[team] += 1
    tot = sum(poss.values()) or 1
    passes = [e for e in events if e.type == "Pass"]
    by_team = lambda et, tm: sum(1 for e in events if e.type == et and e.team == tm)
    comp = lambda tm: sum(1 for e in passes
                          if e.team == tm and e.details.get("outcome") == "complete")
    patt = lambda tm: sum(1 for e in passes if e.team == tm)
    n_frames = len(gs.frames)
    trusted = int(trusted_frame_mask(gs, conf_min).sum())
    return {
        "n_events": len(events),
        "homog_conf_min": conf_min,
        "homography_trusted_pct": round(100 * trusted / max(n_frames, 1), 1),
        "carrier_frames": len(carrier),
        "possession_pct": {0: round(100 * poss[0] / tot, 1),
                           1: round(100 * poss[1] / tot, 1)},
        "passes": {0: patt(0), 1: patt(1)},
        "pass_completion_pct": {
            0: round(100 * comp(0) / max(patt(0), 1), 1),
            1: round(100 * comp(1) / max(patt(1), 1), 1)},
        "carries": {0: by_team("Carry", 0), 1: by_team("Carry", 1)},
        "shots": {0: by_team("Shot", 0), 1: by_team("Shot", 1)},
        "possession_changes": sum(1 for e in events
                                  if e.type == "Possession Change"),
    }


# ── Export (StatsBomb v4-shaped) ─────────────────────────────────────────────
# Follows the StatsBomb Open Data Events structure (doc v4.0.0): common fields
# (id/index/period/timestamp/minute/second/type/possession/possession_team/
# play_pattern/team/player/location/duration), 120x80 coordinates, typed detail
# objects, Ball Receipt* events after completed passes, and a possession
# sequence counter that increments when possession really changes hands.
# Known deviations, declared in meta: coordinates are NOT attack-direction
# normalized (SB flips so each team attacks left→right; we don't yet know
# per-team attack direction), players are track ids (no identities), and shot
# outcomes are "Unknown" (unvalidated).

SB_TYPE = {"Pass": {"id": 30, "name": "Pass"},
           "Carry": {"id": 43, "name": "Carry"},
           "Shot": {"id": 16, "name": "Shot"},
           "Ball Receipt": {"id": 42, "name": "Ball Receipt*"},
           "Ball Recovery": {"id": 2, "name": "Ball Recovery"}}
SB_PLAY_PATTERN = {"id": 1, "name": "Regular Play"}
SB_PASS_OUTCOME = {"incomplete": {"id": 9, "name": "Incomplete"},
                   "interception": {"id": 9, "name": "Incomplete"},
                   "out": {"id": 75, "name": "Out"}}


def _to_sb(xy) -> list:
    """metres 105×68 → StatsBomb 120×80."""
    return [round(xy[0] / PITCH_L * 120.0, 2), round(xy[1] / PITCH_W * 80.0, 2)]


def _sb_team(team: int, slug: str) -> dict:
    return {"id": int(team), "name": f"{slug}-team{team}"}


def _sb_player(tid: int) -> dict:
    return {"id": int(tid), "name": f"track-{tid}"}


def _sb_timestamp(t: float) -> tuple:
    m, s = divmod(max(t, 0.0), 60)
    return f"00:{int(m):02d}:{s:06.3f}", int(m), int(s)


def export_events(events: list[Event], slug: str, summary: dict,
                  fps: float = 25.0) -> Path:
    import uuid
    Config.OUTPUT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # StatsBomb convention: every event's coordinates are given from the
    # acting team's perspective, attacking left→right (toward x=120). Raw
    # pitch coordinates are flipped for the team attacking right-to-left.
    attack_ltr = {int(k): bool(v)
                  for k, v in (summary.get("attack_ltr") or {}).items()}
    gk_tracks = {int(k) for k in (summary.get("gk_tracks") or {})}

    # Real player names when the identity layer has them (src.identity:
    # meta-track consolidation + the naming widget). Falls back to track ids.
    from .identity import load_identity_map
    idmap = load_identity_map(slug) or {}

    def resolve_player(tid: int) -> dict:
        info = idmap.get(int(tid))
        if info:
            name = info.get("name") or f"#{info.get('number')}"
            rec = {"id": int(tid), "name": name}
            if info.get("number") is not None:
                rec["jersey_number"] = int(info["number"])
            return rec
        return _sb_player(tid)

    def norm(xy_sb: list, team: int) -> list:
        if attack_ltr and not attack_ltr.get(int(team), True):
            return [round(120.0 - xy_sb[0], 2), round(80.0 - xy_sb[1], 2)]
        return xy_sb

    out = []
    possession = 1
    possession_team = events[0].team if events else 0

    def base(e, etype, team, player, loc, t=None):
        ts, minute, second = _sb_timestamp(e.time_sec if t is None else t)
        rec = {
            "id": str(uuid.uuid4()), "index": len(out) + 1,
            "period": e.period, "timestamp": ts,
            "minute": minute, "second": second,
            "type": SB_TYPE[etype], "possession": possession,
            "possession_team": _sb_team(possession_team, slug),
            "play_pattern": SB_PLAY_PATTERN,
            "team": _sb_team(team, slug), "player": resolve_player(player),
            "location": norm(_to_sb(loc), team),
            "frame": int(e.frame), "pitch_xy": [round(v, 2) for v in loc],
        }
        if int(player) in gk_tracks:
            rec["position"] = {"id": 1, "name": "Goalkeeper"}
        return rec

    for e in events:
        if e.type == "Possession Change":
            possession += 1
            possession_team = e.team
            rec = base(e, "Ball Recovery", e.team, e.player, e.location)
            out.append(rec)
            continue

        if e.type == "Pass":
            rec = base(e, "Pass", e.team, e.player, e.location)
            end = e.details.get("end_location", list(e.location))
            dx = end[0] - e.location[0]
            dy = end[1] - e.location[1]
            p = {"length": round(float(np.hypot(dx, dy)) / PITCH_L * 120.0, 2),
                 "angle": round(float(np.arctan2(dy, dx)), 3),
                 "end_location": norm(_to_sb(end), e.team)}
            if e.details.get("recipient") is not None:
                p["recipient"] = resolve_player(e.details["recipient"])
            oc = e.details.get("outcome")
            if oc in SB_PASS_OUTCOME:            # complete passes omit outcome
                p["outcome"] = SB_PASS_OUTCOME[oc]
            rec["pass"] = p
            out.append(rec)
            # Ball Receipt for the recipient at the arrival point.
            if oc == "complete" and e.details.get("recipient") is not None:
                r = base(e, "Ball Receipt", e.team, e.details["recipient"],
                         end, t=e.time_sec + 0.4)
                out.append(r)
            continue

        if e.type == "Carry":
            rec = base(e, "Carry", e.team, e.player, e.location)
            rec["carry"] = {"end_location": norm(
                _to_sb(e.details.get("end_location", e.location)), e.team)}
            rec["duration"] = round(e.details.get("length", 0.0) / 4.0, 2)
            out.append(rec)
            continue

        if e.type == "Shot":
            rec = base(e, "Shot", e.team, e.player, e.location)
            rec["shot"] = {"end_location": norm(
                _to_sb(e.details.get("end_location", e.location)), e.team),
                           "outcome": {"id": 0, "name": "Unknown"}}
            out.append(rec)

    payload = {
        "slug": slug,
        "meta": {
            "schema": "statsbomb-events-v4-shaped",
            "coordinates": "120x80, NOT attack-direction normalized",
            "players": "BoT-SORT track ids (no identity mapping yet)",
            "source": "polutka football-computer-vision pipeline",
        },
        "summary": summary,
        "events": out,
    }
    path = Config.OUTPUT_EVENTS_DIR / f"{slug}_events.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser(description="Detect events from a persisted game state")
    ap.add_argument("--match", default="sut-mla")
    args = ap.parse_args()

    gs = GameState.load(args.match)
    events, summary = detect_events(gs)
    path = export_events(events, args.match, summary)

    print(f"\n=== {args.match}: {len(events)} events ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
