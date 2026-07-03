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
# Touch validity (SH golden found the class): a rolling pass that skims past a
# bystander at 5-8 m/s within 3 m mints a phantom touch that SPLITS the real
# pass in two (segE: 43025->43189 became 43025->43203 + 43203->43189). A real
# touch shows at least one of: the ball genuinely AT the feet, a sustained
# hold, or kick evidence on the same player.
TOUCH_MIN_DIST_M = 1.5        # ball must come at least this close once, or
TOUCH_STRONG_FRAMES = 12      # the hold must last this long (0.5 s), or
                              # a detected kick by the same player overlaps
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
# A kick changes velocity, not position: an estimate that teleports between
# consecutive rows is the tracker re-acquiring a different ball (golden segA:
# a 25 m reinit jump from a false mid-pitch ball onto the real throw-in ball
# read as dv≈40 m/s and minted two fake kick-passes). Fastest real ball
# ≈ 40 m/s ≈ 1.6 m/frame; consecutive estimates ≤3 frames apart stay < 5 m.
MAX_KICK_JUMP_M = 5.0
# Restart (throw-in / corner / goal-kick) handling: a slow ball parked on the
# boundary is a restart being set up, not open play (the thrower's ball
# projects 0.1 m INSIDE the line — position-crossing alone misses it).
STICKY_LINE_M = 0.4           # ball within this of a line (or beyond) ...
STICKY_MAX_SPEED = 5.0        # ... and slower than this ...
STICKY_FRAMES = 6             # ... for this many estimates ⇒ dead (restart setup)
MIN_DEAD_RUN_FRAMES = 30      # sticky-triggered dead runs shorter than this
                              # (1.2 s) are homography flicker, not a restart
                              # — a real out-to-restart cycle takes seconds
RESTART_TOUCH_WINDOW_S = 3.0  # first touch within this of resume = the restart
RESTART_THROWER_R_M = 4.0     # restart taker must be within this of the dead ball
RESTART_LINE_MAX_M = 4.0      # restart out-point must be this close to a line
                              # (dead runs with mid-pitch ball estimates are
                              # tracker garbage — no pass synthesis from them)
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
    minted during throw-in retrieval fabricated passes (golden segA).

    Two triggers:
      * position beyond the boundary margin (the original rule), and
      * a slow ball parked ON the line for STICKY_FRAMES — a restart being
        set up. Homography error near the touchline keeps a genuinely-out
        ball's estimate marginally inside (the sut-mla thrower's ball sat
        0.1 m inside), so crossing alone under-triggers.
    """
    dead = set()
    dead_state = False
    entered_by_sticky = False
    run_start = None
    back_in_since = None
    sticky_run = 0
    for r in ball.itertuples(index=False):
        if not (r.bx == r.bx):
            continue
        f = int(r.frame)
        oob = (r.bx < -OUT_MARGIN_M or r.bx > PITCH_L + OUT_MARGIN_M
               or r.by < -OUT_MARGIN_M or r.by > PITCH_W + OUT_MARGIN_M)
        line_dist = min(r.bx, PITCH_L - r.bx, r.by, PITCH_W - r.by)
        slow = not (r.speed == r.speed) or r.speed <= STICKY_MAX_SPEED
        if line_dist <= STICKY_LINE_M and slow:
            sticky_run += 1
        else:
            sticky_run = 0
        if oob or sticky_run >= STICKY_FRAMES:
            if not dead_state:
                dead_state = True
                run_start = f
                entered_by_sticky = not oob
            elif oob:
                entered_by_sticky = False   # genuine crossing upgrades the run
            back_in_since = None
        elif dead_state:
            if line_dist > STICKY_LINE_M:
                if back_in_since is None:
                    back_in_since = f
                elif f - back_in_since >= DEAD_BALL_COOLDOWN:
                    # Sticky-only flickers shorter than a real restart cycle
                    # are homography drift parking the estimate on the line —
                    # retract them so they can't eat real possession frames.
                    if entered_by_sticky and f - run_start < MIN_DEAD_RUN_FRAMES:
                        dead -= set(range(run_start, f + 1))
                    dead_state = False
            else:
                back_in_since = None
        if dead_state:
            dead.add(f)
    return dead


def detect_restarts(ball: pd.DataFrame, dead: set, fps: float = 25.0) -> list:
    """Classify each dead-ball run into a restart: where the ball went out
    decides the type (touchline → throw-in; goal line → corner when play
    resumes near the corner, else goal kick). Returns
    ``[{f_dead0, f_resume, out_xy, resume_xy, type}, ...]`` in frame order."""
    if not dead:
        return []
    pos = {int(r.frame): (r.bx, r.by) for r in ball.itertuples(index=False)
           if r.bx == r.bx}
    frames = sorted(dead)
    runs = []
    run_start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f - prev > int(fps):          # >1 s gap between dead frames = new run
            runs.append((run_start, prev))
            run_start = f
        prev = f
    runs.append((run_start, prev))

    out = []
    for f0, f1 in runs:
        # Out-point: the estimate closest to a boundary around the dead-run
        # start (a median over the whole run is poisoned by the tracker
        # following wrong balls mid-run — measured on golden segA, where the
        # median sat 30 m infield and named a random player as the taker).
        window = [pos[f] for f in range(f0 - 10, f0 + int(fps)) if f in pos]
        if not window:
            continue
        ox, oy = min(window, key=lambda p: min(p[0], PITCH_L - p[0],
                                               p[1], PITCH_W - p[1]))
        line_dist = min(ox, PITCH_L - ox, oy, PITCH_W - oy)
        if line_dist > RESTART_LINE_MAX_M:
            continue        # no credible out-point — suppress, don't guess
        after = [pos[f] for f in range(f1 + 1, f1 + 1 + int(3 * fps))
                 if f in pos]
        rx, ry = (after[0] if after else (ox, oy))
        # Which boundary was crossed / camped on?
        d_touch = min(oy, PITCH_W - oy)
        d_goal = min(ox, PITCH_L - ox)
        if d_touch <= d_goal:
            rtype = "throw_in"
        else:
            near_corner = (min(ry, PITCH_W - ry) < 8.0
                           and min(rx, PITCH_L - rx) < 8.0)
            rtype = "corner" if near_corner else "goal_kick"
        out.append({"f_dead0": int(f0), "f_resume": int(f1) + 1,
                    "out_xy": (float(ox), float(oy)),
                    "resume_xy": (float(rx), float(ry)),
                    "type": rtype})
    return out


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
        # Both anchors must be real detections: a hindsight bridge is a
        # straight line, so any velocity step at its SEAM with a detected
        # segment is an estimation artifact, not ball contact (SH golden:
        # seam-kicks mid-flight minted phantom touches for bystanders and
        # split real passes in two).
        if r1.source != "detected" or r0.source != "detected":
            continue
        # A kick changes velocity, not position: a teleporting estimate is
        # the tracker re-acquiring a different ball, not ball contact.
        if float(np.hypot(r1.bx - r0.bx, r1.by - r0.by)) > MAX_KICK_JUMP_M:
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
    ball_pos = {}
    if ball is not None:
        detected_frames = set(
            ball.loc[ball["source"] == "detected", "frame"].astype(int))
        ball_pos = {int(r.frame): (r.bx, r.by)
                    for r in ball.itertuples(index=False) if r.bx == r.bx}
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
        # Touch validity: close approach, sustained hold, or kick evidence —
        # otherwise it's a rolling ball skimming past a bystander.
        if (f1 - f0 + 1) < TOUCH_STRONG_FRAMES and ball_pos:
            dmin = np.inf
            for f in range(int(f0), int(f1) + 1):
                bp = ball_pos.get(f)
                pp = pos.get((f, tid))
                if bp and pp:
                    dmin = min(dmin, float(np.hypot(pp[0] - bp[0],
                                                    pp[1] - bp[1])))
            kicked = kicks and any(
                k[1] == tid and f0 - 3 <= k[0] <= f1 + 3 for k in kicks)
            if dmin > TOUCH_MIN_DIST_M and not kicked:
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


PATTERN_OF_RESTART = {"throw_in": "From Throw In", "corner": "From Corner",
                      "goal_kick": "From Goal Kick"}


def synthesize_restart_passes(gs: GameState, restarts: list,
                              touches: list, events: list,
                              team_override: Optional[dict],
                              fps: float) -> list:
    """One Pass event per restart: the taker (nearest player to the dead
    ball as play resumes) to the first controlled touch. StatsBomb models
    throw-ins/corners/goal kicks as passes; the ordinary carrier logic can't
    see them because the taker's touch happens inside the dead-ball window
    (golden segA throw-in was a measured FN). Skipped when the ordinary
    logic already produced a pass by the same player nearby in time."""
    pl = gs.players[gs.players.pitch_x.notna()]
    if team_override:
        pl = pl.copy()
        eff = pl["track_id"].map(team_override)
        pl["team_id"] = eff.fillna(pl["team_id"]).astype(int)
    pl = pl[pl.team_id.isin([0, 1])]
    by_frame: dict[int, list] = defaultdict(list)
    for r in pl.itertuples(index=False):
        by_frame[int(r.frame)].append((int(r.track_id), int(r.team_id),
                                       r.pitch_x, r.pitch_y))

    out = []
    for rs in restarts:
        resume = rs["f_resume"]
        window = int(RESTART_TOUCH_WINDOW_S * fps)
        cands = [t for t in touches
                 if resume - 2 <= t.f_start <= resume + window]
        if not cands:
            continue
        first = min(cands, key=lambda t: t.f_start)
        # The taker: nearest player to the dead-ball spot as play resumes.
        best, best_d = None, np.inf
        for f in range(resume - 12, resume + 3):
            for tid, team, px, py in by_frame.get(f, []):
                d = float(np.hypot(px - rs["out_xy"][0], py - rs["out_xy"][1]))
                if d < best_d:
                    best_d, best = d, (tid, team)
        if best is None or best_d > RESTART_THROWER_R_M:
            continue
        taker, taker_team = best
        if taker == first.track_id:
            continue        # taker keeps it (short corner / quick dribble)
        # The ordinary pipeline may already have caught this pass.
        already = any(e.type == "Pass" and e.player == taker
                      and abs(e.frame - resume) <= window for e in events)
        if already:
            continue
        complete = first.team == taker_team
        details = {
            "end_location": list(first.xy_start),
            "outcome": "complete" if complete else "incomplete",
            "length": round(float(np.hypot(
                first.xy_start[0] - rs["out_xy"][0],
                first.xy_start[1] - rs["out_xy"][1])), 2),
            "play_pattern": PATTERN_OF_RESTART[rs["type"]],
            "restart": rs["type"],
        }
        if complete:
            details["recipient"] = int(first.track_id)
        out.append(Event(0, int(gs.meta.get("period", 1)),
                         int(resume - 1), _tsec(gs, resume - 1), "Pass",
                         int(taker_team), int(taker),
                         tuple(rs["out_xy"]), details))
    return out


def tag_play_patterns(events: list, restarts: list, gs: GameState) -> None:
    """Assign StatsBomb ``play_pattern`` to every event: a restart's pattern
    holds until the next real possession change (which resets to Regular
    Play, matching SB semantics where a recovery starts a regular-play
    possession). The first events of an artifact that begins at kickoff get
    From Kick Off."""
    starts_at_kickoff = (len(gs.frames) > 0
                         and float(gs.frames["time_sec"].iloc[0]) < 10.0)
    current = "From Kick Off" if starts_at_kickoff else "Regular Play"
    ri = 0
    rs = sorted(restarts, key=lambda r: r["f_resume"])
    for e in sorted(events, key=lambda e: e.frame):
        while ri < len(rs) and rs[ri]["f_resume"] <= e.frame:
            current = PATTERN_OF_RESTART[rs[ri]["type"]]
            ri += 1
        if e.type == "Possession Change":
            current = "Regular Play"
        e.details.setdefault("play_pattern", current)


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

    # Restarts: synthesize the taker's pass (throw-in/corner/goal kick) and
    # tag every event's play_pattern.
    restarts = detect_restarts(ball, dead, fps)
    events.extend(synthesize_restart_passes(gs, restarts, touches, events,
                                            gk_map, fps))
    tag_play_patterns(events, restarts, gs)

    events.sort(key=lambda e: (e.frame, e.index))
    for k, e in enumerate(events):
        e.index = k
    summary = _summary(events, carrier, gs, conf_min)
    summary["attack_ltr"] = {int(k): bool(v)
                             for k, v in directions.attack_ltr.items()}
    summary["attack_direction_confidence"] = round(directions.confidence, 3)
    summary["gk_tracks"] = {int(k): int(v) for k, v in gk_map.items()}
    summary["restarts"] = {t: sum(1 for r in restarts if r["type"] == t)
                           for t in ("throw_in", "corner", "goal_kick")}
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
        "n_frames": n_frames,
        "period": int(gs.meta.get("period", 1)),
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
# play_pattern/team/player/location/duration), 120x80 coordinates
# (attack-direction normalized: each team attacks left→right), typed detail
# objects, Ball Receipt* events after completed passes, and a possession
# sequence counter that increments when possession really changes hands —
# continuing across halves in the match-level export.
# Known deviations, declared in meta: players are track ids unless an identity
# file names them, and track ids are per-period (BoT-SORT restarts each run,
# so period-2 ids are namespaced by PERIOD_TID_OFFSET until cross-half ReID).

SB_TYPE = {"Pass": {"id": 30, "name": "Pass"},
           "Carry": {"id": 43, "name": "Carry"},
           "Shot": {"id": 16, "name": "Shot"},
           "Ball Receipt": {"id": 42, "name": "Ball Receipt*"},
           "Ball Recovery": {"id": 2, "name": "Ball Recovery"}}
SB_PLAY_PATTERNS = {
    "Regular Play":   {"id": 1, "name": "Regular Play"},
    "From Corner":    {"id": 2, "name": "From Corner"},
    "From Free Kick": {"id": 3, "name": "From Free Kick"},
    "From Throw In":  {"id": 4, "name": "From Throw In"},
    "From Counter":   {"id": 6, "name": "From Counter"},
    "From Goal Kick": {"id": 7, "name": "From Goal Kick"},
    "From Keeper":    {"id": 8, "name": "From Keeper"},
    "From Kick Off":  {"id": 9, "name": "From Kick Off"},
}
SB_PLAY_PATTERN = SB_PLAY_PATTERNS["Regular Play"]
SB_PASS_OUTCOME = {"incomplete": {"id": 9, "name": "Incomplete"},
                   "interception": {"id": 9, "name": "Incomplete"},
                   "out": {"id": 75, "name": "Out"}}
SB_SHOT_OUTCOME = {"goal": {"id": 97, "name": "Goal"}}   # else Unknown

# Track-id namespace stride between periods in match-level exports. Raw
# BoT-SORT ids restart at 1 for every artifact run, so period-1 track 20 and
# period-2 track 20 are different people; a merged file must not conflate them.
PERIOD_TID_OFFSET = 100000


def _to_sb(xy) -> list:
    """metres 105×68 → StatsBomb 120×80."""
    return [round(xy[0] / PITCH_L * 120.0, 2), round(xy[1] / PITCH_W * 80.0, 2)]


def _sb_team(team: int, slug: str) -> dict:
    return {"id": int(team), "name": f"{slug}-team{team}"}


def _sb_timestamp(t: float) -> tuple:
    m, s = divmod(max(t, 0.0), 60)
    return f"00:{int(m):02d}:{s:06.3f}", int(m), int(s)


def _sb_records_for_period(events: list[Event], slug: str, summary: dict,
                           index_start: int = 0, possession: int = 1,
                           possession_team: Optional[int] = None,
                           idmap: Optional[dict] = None,
                           meta_map: Optional[dict] = None) -> tuple:
    """SB records for one period's events.

    Continuation state (``index_start`` / ``possession`` / ``possession_team``)
    lets the match-level export chain periods with one running index and a
    possession counter that survives halftime. Track ids are namespaced by
    period (period 1 keeps raw ids). ``meta_map`` (track→meta-track from
    :func:`src.identity.meta_map`) keys player stats to consolidated players
    instead of ephemeral tracks — without it a match's "top passer" is
    whichever fragment of a player kept one BoT-SORT id longest. Returns
    ``(records, possession, possession_team)``.
    """
    import uuid
    period = events[0].period if events else 1
    minute_offset = 45 * (period - 1)   # SB: minute continues across halves
    attack_ltr = {int(k): bool(v)
                  for k, v in (summary.get("attack_ltr") or {}).items()}
    gk_tracks = {int(k): int(v)
                 for k, v in (summary.get("gk_tracks") or {}).items()}
    idmap = idmap or {}

    def ns(tid: int) -> int:
        return int(tid) + PERIOD_TID_OFFSET * (period - 1)

    # Goalkeepers get a period-independent identity: the roles signature is
    # positionally certain, so team T's GK is the same player in both halves
    # (barring a GK substitution) — the one cross-half merge we can make
    # honestly. Outfield cross-half ReID was measured and rejected: the
    # team-classifier embeddings carry no within-team identity signal
    # (P(same-player pair closer) = 0.43, i.e. chance).
    GK_ID_BASE = 900000

    def resolve_player(tid: int) -> dict:
        if int(tid) < 0:            # oracle goals have no identified scorer
            return {"id": -1, "name": "unknown"}
        info = idmap.get(int(tid))
        suffix = "" if period == 1 else f"-h{period}"
        if info:
            name = info.get("name") or f"#{info.get('number')}"
            rec = {"id": ns(tid), "name": name}
            if info.get("number") is not None:
                rec["jersey_number"] = int(info["number"])
            return rec
        if int(tid) in gk_tracks:
            team = gk_tracks[int(tid)]
            return {"id": GK_ID_BASE + team, "name": f"goalkeeper-t{team}"}
        if meta_map is not None:
            mid = int(meta_map.get(int(tid), tid))
            return {"id": ns(mid), "name": f"player-m{mid}{suffix}"}
        return {"id": ns(tid), "name": f"track-{tid}{suffix}"}

    # StatsBomb convention: every event's coordinates are given from the
    # acting team's perspective, attacking left→right (toward x=120). Raw
    # pitch coordinates are flipped for the team attacking right-to-left.
    def norm(xy_sb: list, team: int) -> list:
        if attack_ltr and not attack_ltr.get(int(team), True):
            return [round(120.0 - xy_sb[0], 2), round(80.0 - xy_sb[1], 2)]
        return xy_sb

    out = []
    if possession_team is None:
        possession_team = events[0].team if events else 0

    def base(e, etype, team, player, loc, t=None):
        ts, minute, second = _sb_timestamp(e.time_sec if t is None else t)
        rec = {
            "id": str(uuid.uuid4()), "index": index_start + len(out) + 1,
            "period": e.period, "timestamp": ts,
            "minute": minute + minute_offset, "second": second,
            "type": SB_TYPE[etype], "possession": possession,
            "possession_team": _sb_team(possession_team, slug),
            "play_pattern": SB_PLAY_PATTERNS.get(
                e.details.get("play_pattern", "Regular Play"),
                SB_PLAY_PATTERN),
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
            oc = e.details.get("outcome")
            rec["shot"] = {"end_location": norm(
                _to_sb(e.details.get("end_location", e.location)), e.team),
                           "outcome": SB_SHOT_OUTCOME.get(
                               oc, {"id": 0, "name": "Unknown"})}
            if e.details.get("goal_oracle"):
                rec["shot"]["oracle"] = e.details["goal_oracle"]
            out.append(rec)

    return out, possession, possession_team


def _sb_meta(slug: str, periods: list) -> dict:
    return {
        "schema": "statsbomb-events-v4-shaped",
        "coordinates": "120x80, attack-direction normalized (left->right "
                       "per acting team)",
        "players": "BoT-SORT track ids unless named via data/identities; "
                   f"period-2 ids offset by {PERIOD_TID_OFFSET} (no "
                   "cross-half ReID yet)",
        "periods": periods,
        "source": "polutka football-computer-vision pipeline",
    }


def export_events(events: list[Event], slug: str, summary: dict,
                  fps: float = 25.0, meta_map: Optional[dict] = None) -> Path:
    """Single-period export → ``output/events/{slug}_p{N}_events.json``."""
    from .identity import load_identity_map
    Config.OUTPUT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    period = events[0].period if events else 1
    recs, _, _ = _sb_records_for_period(
        events, slug, summary, idmap=load_identity_map(slug, period),
        meta_map=meta_map)
    payload = {
        "slug": slug,
        "meta": _sb_meta(slug, [period]),
        "summary": summary,
        "events": recs,
    }
    path = Config.OUTPUT_EVENTS_DIR / f"{slug}_p{period}_events.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _match_summary(halves: list) -> dict:
    """Aggregate per-period summaries into match-level numbers. Possession is
    weighted by each period's carrier-frame count."""
    all_events = [e for events, _, _ in halves for e in events]
    poss_frames = {0: 0.0, 1: 0.0}
    trusted_f = total_f = 0
    for _, s, _ in halves:
        cf = s.get("carrier_frames", 0)
        pp = s.get("possession_pct", {})
        for t in (0, 1):
            poss_frames[t] += pp.get(t, 0.0) * cf / 100.0
        total_f += s.get("n_frames", 0)
        trusted_f += s.get("n_frames", 0) * s.get("homography_trusted_pct", 0) / 100.0
    tot = sum(poss_frames.values()) or 1
    passes = [e for e in all_events if e.type == "Pass"]
    patt = lambda tm: sum(1 for e in passes if e.team == tm)
    comp = lambda tm: sum(1 for e in passes if e.team == tm
                          and e.details.get("outcome") == "complete")
    by_team = lambda et, tm: sum(1 for e in all_events
                                 if e.type == et and e.team == tm)
    goals = lambda tm: sum(1 for e in all_events if e.type == "Shot"
                           and e.team == tm
                           and e.details.get("outcome") == "goal")
    return {
        "n_events": len(all_events),
        "periods": [p for _, _, p in halves],
        "possession_pct": {0: round(100 * poss_frames[0] / tot, 1),
                           1: round(100 * poss_frames[1] / tot, 1)},
        "passes": {0: patt(0), 1: patt(1)},
        "pass_completion_pct": {
            0: round(100 * comp(0) / max(patt(0), 1), 1),
            1: round(100 * comp(1) / max(patt(1), 1), 1)},
        "carries": {0: by_team("Carry", 0), 1: by_team("Carry", 1)},
        "shots": {0: by_team("Shot", 0), 1: by_team("Shot", 1)},
        "goals": {0: goals(0), 1: goals(1)},
        "possession_changes": sum(1 for e in all_events
                                  if e.type == "Possession Change"),
        "homography_trusted_pct": round(100 * trusted_f / max(total_f, 1), 1),
    }


def inject_oracle_goals(slug: str, halves: list) -> int:
    """Fold scoreboard goal-oracle goals (:mod:`src.score_ocr`) into the
    per-period event lists as Shot events with outcome ``goal``.

    The oracle knows a goal happened (score change on the broadcast graphics)
    but not the shot's pixel-precise moment or location: the event is anchored
    at the first frame the new score is visible (within seconds-to-a-minute
    after the real goal, bracket declared in ``details.goal_oracle``) and
    placed at the attacked goal's penalty spot, flagged ``location_estimated``.
    Returns the number of goals injected."""
    from .score_ocr import oracle_path
    p = oracle_path(slug)
    if not p.exists():
        return 0
    oracle = json.loads(p.read_text())
    home = oracle.get("home_team_id")
    if home is None:
        print("goal-oracle present but home_team_id unknown - goals not "
              "injected (re-run src.score_ocr with --home_team)")
        return 0

    by_period = {period: (events, summary) for events, summary, period in halves}
    n = 0
    for g in oracle.get("goals", []):
        period = int(g["period"])
        if period not in by_period:
            continue
        events, summary = by_period[period]
        team = int(home) if g["scorer_side"] == "home" else 1 - int(home)
        attack_ltr = (summary.get("attack_ltr") or {}).get(team, True)
        spot = (PITCH_L - 11.0, PITCH_W / 2) if attack_ltr else (11.0, PITCH_W / 2)
        goal_line = (PITCH_L, PITCH_W / 2) if attack_ltr else (0.0, PITCH_W / 2)
        e = Event(
            index=len(events), period=period,
            frame=int(g.get("anchor_frame", g["first_seen_frame"])),
            time_sec=float(g["time_sec"]), type="Shot", team=team,
            player=-1, location=spot,
            details={
                "outcome": "goal",
                "end_location": list(goal_line),
                "location_estimated": True,
                "goal_oracle": {
                    "score_after": g["score_after"],
                    "bracket_frames": [g["last_old_score_frame"],
                                       g["first_seen_frame"]],
                    "source": g["first_seen_source"],
                },
            })
        events.append(e)
        events.sort(key=lambda ev: (ev.frame, ev.index))
        n += 1
    return n


def export_match_events(slug: str, halves: list,
                        meta_maps: Optional[dict] = None) -> Path:
    """Match-level export: ``halves`` is ``[(events, summary, period), ...]``
    → one ``output/events/{slug}_events.json`` with a running event index,
    possession numbering that continues across halves, and per-period
    summaries preserved under ``summary.periods_detail``. ``meta_maps``:
    {period: track→meta map} for player consolidation."""
    from .identity import load_identity_map
    Config.OUTPUT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    halves = sorted(halves, key=lambda h: h[2])

    recs: list = []
    possession, possession_team = 1, None
    for i, (events, summary, period) in enumerate(halves):
        if i > 0:
            # The period kickoff starts a fresh possession; we don't detect
            # the kickoff itself, so the first event's team is the best proxy.
            possession += 1
            possession_team = events[0].team if events else possession_team
        r, possession, possession_team = _sb_records_for_period(
            events, slug, summary, index_start=len(recs),
            possession=possession, possession_team=possession_team,
            idmap=load_identity_map(slug, period),
            meta_map=(meta_maps or {}).get(period))
        recs.extend(r)

    match_summary = _match_summary(halves)
    match_summary["periods_detail"] = {p: s for _, s, p in halves}

    # Teams swap ends at halftime — if inferred attack directions do NOT flip
    # between periods, one period's roles inference is wrong.
    if len(halves) >= 2:
        d1 = halves[0][1].get("attack_ltr") or {}
        d2 = halves[1][1].get("attack_ltr") or {}
        flipped = all(bool(d1.get(t, d1.get(str(t)))) !=
                      bool(d2.get(t, d2.get(str(t)))) for t in (0, 1)
                      if (t in d1 or str(t) in d1) and (t in d2 or str(t) in d2))
        match_summary["attack_direction_flipped_at_halftime"] = bool(flipped)
        if not flipped:
            print("WARNING: attack directions did not flip between halves - "
                  "check roles inference on one of the artifacts")

    payload = {
        "slug": slug,
        "meta": _sb_meta(slug, [p for _, _, p in halves]),
        "summary": match_summary,
        "events": recs,
    }
    path = Config.OUTPUT_EVENTS_DIR / f"{slug}_events.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    from .game_state import available_periods
    from .identity import meta_map as build_meta_map
    ap = argparse.ArgumentParser(description="Detect events from a persisted game state")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--half", type=int, default=None, choices=[1, 2],
                    help="export just this period; default: all stored "
                         "periods merged into one match-level file")
    ap.add_argument("--raw_tracks", action="store_true",
                    help="skip meta-track consolidation of player ids")
    args = ap.parse_args()

    if args.half is not None:
        gs = GameState.load(args.match, period=args.half)
        events, summary = detect_events(gs)
        n_goals = inject_oracle_goals(args.match, [(events, summary, args.half)])
        if n_goals:
            print(f"[p{args.half}] {n_goals} oracle goal(s) injected")
        mm = None if args.raw_tracks else build_meta_map(gs)
        path = export_events(events, args.match, summary, meta_map=mm)
        print(f"\n=== {args.match} p{args.half}: {len(events)} events ===")
        print(json.dumps(summary, indent=2))
        print(f"Saved -> {path}")
        return

    halves = []
    meta_maps = {}
    for p in available_periods(args.match):
        gs = GameState.load(args.match, period=p)
        events, summary = detect_events(gs)
        halves.append((events, summary, p))
        if not args.raw_tracks:
            meta_maps[p] = build_meta_map(gs)
        print(f"[p{p}] {len(events)} events, "
              f"possession {summary['possession_pct']}")
    n_goals = inject_oracle_goals(args.match, halves)
    if n_goals:
        print(f"{n_goals} oracle goal(s) injected")
    path = export_match_events(args.match, halves, meta_maps=meta_maps)
    payload = json.loads(path.read_text())
    print(f"\n=== {args.match}: {payload['summary']['n_events']} events "
          f"across periods {payload['summary']['periods']} ===")
    print(json.dumps({k: v for k, v in payload["summary"].items()
                      if k != "periods_detail"}, indent=2))
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
