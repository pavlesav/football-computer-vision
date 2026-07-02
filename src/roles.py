"""
Team attack directions and goalkeeper identification from the game state.

Both are inferred from the persisted artifact alone (no video, no GPU), and
both unlock things the event layer measured as missing:

* **Attack direction** — without knowing which goal a team attacks, StatsBomb
  coordinates cannot be attack-normalized, "final third" is undefined, and the
  shot rule has to guess the target goal ("whichever end is nearer"), which is
  wrong for shots from own-half clearances and weakens shot precision.
* **Goalkeepers** — the team classifier labels GKs 'Other' (their kit matches
  neither team), which made them invisible to possession: the golden set's GK
  build-up pass is a measured false negative. Identifying GK tracks by
  positional signature and assigning them to the team defending that goal
  makes GK touches, goal kicks, and build-up passes first-class events.

Method notes
------------
Attack direction uses defensive-shape evidence, not GKs (avoids circularity):
on trusted frames where the ball is deep in one half, the *defending* team has
systematically more outfield players in that deep zone (they retreat behind
the ball; attackers hold higher lines). Averaged over hundreds of deep-ball
frames per half this is a large-margin signal, robust to team-label noise.

Goalkeepers are tracks that live in the goal zone and never leave it: median
pitch-x inside the defensive ~18m, high-percentile x still deep (outfielders
visit deep zones; only GKs *stay*), central y, and enough lifetime to rule out
detection flickers. Referees never satisfy the stay-deep constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .game_state import GameState, trusted_frame_mask

PITCH_L, PITCH_W = 105.0, 68.0

DEEP_ZONE_M = 30.0          # "ball is deep" / defender-count zone from a goal line
GK_MEDIAN_X_M = 18.0        # GK median x within this of the goal line
GK_STAY_X_M = 32.0          # GK's far-percentile x still within this (stays deep)
GK_Y_MIN, GK_Y_MAX = 18.0, 50.0
GK_MIN_FRAMES = 40
# The decisive GK discriminator: in frames where he appears, the goalkeeper is
# almost always THE deepest player on the pitch (nobody stands behind him).
# Outfield defenders parked deep during a siege fail this — the GK is behind
# them. Without it, a full half produced ~200 "GK" tracks (entire defensive
# lines during sustained pressure).
GK_DEEPEST_FRAC = 0.7
GK_DEEPEST_TOL_M = 1.5      # within this of the frame's extreme still counts


@dataclass
class TeamDirections:
    """Attack directions for one period. ``attack_ltr[team]`` is True when the
    team attacks left-to-right (toward x=105) in raw pitch coordinates."""
    attack_ltr: dict
    confidence: float
    n_deep_frames: int

    def goal_x(self, team: int) -> float:
        """x of the goal ``team`` attacks (raw coords)."""
        return PITCH_L if self.attack_ltr.get(team, True) else 0.0

    def defends_left(self, team: int) -> bool:
        return self.attack_ltr.get(team, True)


def infer_attack_direction(gs: GameState,
                           conf_min: Optional[float] = None) -> TeamDirections:
    """Defensive-shape vote over trusted deep-ball frames."""
    trusted = set(gs.frames.loc[trusted_frame_mask(gs, conf_min),
                                "frame"].astype(int))
    f = gs.frames
    ball = f.loc[f["frame"].isin(trusted) & f["ball_pitch_x"].notna(),
                 ["frame", "ball_pitch_x"]]
    pl = gs.players
    pl = pl[pl.team_id.isin([0, 1]) & pl.pitch_x.notna()
            & pl.frame.isin(trusted)]
    counts = pl.groupby(["frame", "team_id"]).agg(
        n=("track_id", "size"),
        deep_l=("pitch_x", lambda x: (x < DEEP_ZONE_M).sum()),
        deep_r=("pitch_x", lambda x: (x > PITCH_L - DEEP_ZONE_M).sum()),
    ).reset_index()
    merged = counts.merge(ball, on="frame")

    # Frames with the ball deep left: fraction of each team's visible players
    # that are also deep left. Defenders retreat; attackers hold the line.
    evid = {0: 0.0, 1: 0.0}
    n_deep = 0
    for side, col in (("l", "deep_l"), ("r", "deep_r")):
        if side == "l":
            sel = merged[merged.ball_pitch_x < DEEP_ZONE_M]
        else:
            sel = merged[merged.ball_pitch_x > PITCH_L - DEEP_ZONE_M]
        if sel.empty:
            continue
        frac = sel.assign(fr=sel[col] / sel["n"].clip(lower=1)) \
                  .groupby("team_id")["fr"].mean()
        n_deep += sel["frame"].nunique()
        if 0 in frac.index and 1 in frac.index:
            # positive → team0 defends this side more than team1
            margin = float(frac[0] - frac[1])
            evid[0] += margin if side == "l" else -margin
            evid[1] -= margin if side == "l" else -margin

    # evid[t] > 0 → team t defends LEFT → attacks left-to-right? No:
    # defending left means their goal is at x=0, so they attack toward 105.
    attack_ltr = {0: evid[0] > 0, 1: evid[1] > 0}
    if attack_ltr[0] == attack_ltr[1]:
        # Degenerate (should not happen with the symmetric evidence): fall
        # back to mean-position asymmetry.
        mean_x = pl.groupby("team_id")["pitch_x"].mean()
        attack_ltr = {0: mean_x.get(0, 50) < mean_x.get(1, 50),
                      1: mean_x.get(1, 50) < mean_x.get(0, 50)}
    confidence = float(min(abs(evid[0]), abs(evid[1])))
    return TeamDirections(attack_ltr=attack_ltr, confidence=confidence,
                          n_deep_frames=n_deep)


def identify_goalkeepers(gs: GameState,
                         directions: Optional[TeamDirections] = None) -> dict:
    """{track_id: team} for tracks with a goalkeeper positional signature.

    A GK track lives inside one goal zone and stays there; it is assigned to
    the team *defending* that goal. Works for tracks of any classifier label
    (GK kits are usually classified 'Other', but a GK mislabeled as an
    outfield team still gets the positionally-correct team here).
    """
    if directions is None:
        directions = infer_attack_direction(gs)
    pl = gs.players[gs.players.pitch_x.notna()].copy()

    # Frame extremes: is this player (nearly) the deepest on the pitch?
    ext = pl.groupby("frame")["pitch_x"].agg(["min", "max"]).rename(
        columns={"min": "fx_min", "max": "fx_max"})
    pl = pl.merge(ext, left_on="frame", right_index=True)
    pl["deepest_l"] = pl["pitch_x"] <= pl["fx_min"] + GK_DEEPEST_TOL_M
    pl["deepest_r"] = pl["pitch_x"] >= pl["fx_max"] - GK_DEEPEST_TOL_M

    stats = pl.groupby("track_id").agg(
        n=("frame", "size"),
        med_x=("pitch_x", "median"),
        med_y=("pitch_y", "median"),
        far_x_left=("pitch_x", lambda x: x.quantile(0.9)),
        far_x_right=("pitch_x", lambda x: x.quantile(0.1)),
        deepest_l=("deepest_l", "mean"),
        deepest_r=("deepest_r", "mean"),
    )

    labels = gs.players.groupby("track_id")["team_id"].first()

    out = {}
    for tid, r in stats.iterrows():
        if r.n < GK_MIN_FRAMES or not (GK_Y_MIN <= r.med_y <= GK_Y_MAX):
            continue
        if (r.med_x <= GK_MEDIAN_X_M and r.far_x_left <= GK_STAY_X_M
                and r.deepest_l >= GK_DEEPEST_FRAC):
            goal_left = True
        elif (r.med_x >= PITCH_L - GK_MEDIAN_X_M
                and r.far_x_right >= PITCH_L - GK_STAY_X_M
                and r.deepest_r >= GK_DEEPEST_FRAC):
            goal_left = False
        else:
            continue
        team = 0 if directions.defends_left(0) == goal_left else 1
        # Overriding a confident outfield label needs extreme depth: a short
        # defender/striker track can satisfy the ordinary gates during one
        # deep spell, but nobody except the GK *lives* on the goal line.
        lab = labels.get(tid)
        if lab in (0, 1) and lab != team:
            deep_enough = (r.med_x <= 12.0) if goal_left \
                else (r.med_x >= PITCH_L - 12.0)
            if not deep_enough:
                continue
        out[int(tid)] = team
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Infer attack directions + goalkeepers")
    ap.add_argument("--match", default="sut-mla")
    args = ap.parse_args()
    gs = GameState.load(args.match)
    d = infer_attack_direction(gs)
    print(f"attack left-to-right: team0={d.attack_ltr[0]} team1={d.attack_ltr[1]} "
          f"(confidence {d.confidence:.3f}, {d.n_deep_frames} deep-ball frames)")
    gks = identify_goalkeepers(gs, d)
    pl = gs.players
    for tid, team in sorted(gks.items()):
        sub = pl[pl.track_id == tid]
        print(f"  GK track {tid}: assigned team {team} | frames {len(sub)} | "
              f"median x {sub.pitch_x.median():.1f} | classifier label "
              f"{int(sub.team_id.iloc[0])}")


if __name__ == "__main__":
    main()
