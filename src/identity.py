"""
Player identity: consolidate ephemeral tracks into per-player meta-tracks and
map them to real names.

The problem: BoT-SORT produces thousands of short track ids per match (every
camera cut kills them), so "pass by track-20252" is useless to a coach. The
budget solution (roadmap items 4+5) is:

1. **Consolidation** (automatic): merge tracks into *meta-tracks* using team
   identity + kinematic continuity — a track that starts moments after another
   ends, close enough that a player could have walked there, on the same team,
   is the same player. Goalkeeper tracks consolidate trivially via
   :mod:`src.roles` (all left-goal GK tracks are one person). Appearance
   embeddings can tighten this later; the pipeline does not persist them yet.
2. **Naming** (human, ~10-15 min/match): an ipywidgets gallery shows each
   meta-track's best crops; the analyst types the shirt number / name from the
   public lineup. One entry names every constituent track. Saved to
   ``data/identities/{slug}.json`` and picked up automatically by the
   StatsBomb export.

Consolidation is deliberately conservative: a wrong merge poisons two players'
stats, a missed merge just leaves two rows for one player. Gates: same
effective team, temporal gap ≤ ``max_gap_s``, spatial gap ≤ what a player can
cover in that time (plus a margin for re-detection jitter), and no temporal
overlap (a player cannot be in two tracks at once).

Usage::

    python -m src.identity --match sut-mla            # consolidation report
    # notebook:
    from src.identity import consolidate_tracks, build_identity_widget
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .roles import infer_attack_direction, identify_goalkeepers

MAX_GAP_S = 10.0          # max unseen time to still merge two tracks
MAX_SPEED_MS = 8.0        # players cover at most this while unseen
SLACK_M = 6.0             # margin for detection/projection jitter at endpoints
MIN_TRACK_FRAMES = 10     # ignore flicker tracks entirely


def identities_dir() -> Path:
    d = Config.PROJECT_ROOT / "data" / "identities"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Consolidation ────────────────────────────────────────────────────────────

def _track_summaries(gs, gk_map: dict) -> pd.DataFrame:
    pl = gs.players[gs.players.pitch_x.notna()].copy()
    eff = pl["track_id"].map(gk_map)
    pl["eff_team"] = eff.fillna(pl["team_id"]).astype(int)

    def first_last(g):
        g = g.sort_values("frame")
        return pd.Series({
            "team": int(g["eff_team"].mode().iat[0]),
            "f0": int(g["frame"].iat[0]), "f1": int(g["frame"].iat[-1]),
            "x0": float(g["pitch_x"].iat[0]), "y0": float(g["pitch_y"].iat[0]),
            "x1": float(g["pitch_x"].iat[-1]), "y1": float(g["pitch_y"].iat[-1]),
            "n": len(g),
        })

    s = pl.groupby("track_id").apply(first_last, include_groups=False)
    s = s[s["n"] >= MIN_TRACK_FRAMES]
    return s.reset_index()


def consolidate_tracks(gs, fps: Optional[float] = None,
                       max_gap_s: float = MAX_GAP_S,
                       max_speed: float = MAX_SPEED_MS) -> pd.DataFrame:
    """Returns a DataFrame ``track_id, meta_id, team, frames`` mapping every
    substantial track to a meta-track (one player, ideally)."""
    fps = fps or gs.fps
    directions = infer_attack_direction(gs)
    gk_map = identify_goalkeepers(gs, directions)
    s = _track_summaries(gs, gk_map)

    # Union-find over tracks.
    parent = {int(t): int(t) for t in s["track_id"]}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Goalkeepers: every GK track of the same goal is the same person.
    for team in (0, 1):
        gks = [t for t, tm in gk_map.items() if tm == team and t in parent]
        for a, b in zip(gks, gks[1:]):
            union(a, b)

    # Kinematic continuity, greedy in time order: each starting track claims
    # the best-matching ended track; each ended track is claimed once.
    s = s.sort_values("f0").reset_index(drop=True)
    ends = []      # (f1, x1, y1, tid, team) of tracks already ended, unclaimed
    max_gap_f = int(max_gap_s * fps)
    for r in s.itertuples(index=False):
        # release nothing; scan candidates
        best, best_cost = None, np.inf
        for i, (f1, x1, y1, tid, team) in enumerate(ends):
            gap = r.f0 - f1
            if gap <= 0 or gap > max_gap_f:
                continue
            if team != r.team:
                continue
            d = float(np.hypot(r.x0 - x1, r.y0 - y1))
            if d > max_speed * gap / fps + SLACK_M:
                continue
            cost = d + gap / fps        # prefer near in space, then in time
            if cost < best_cost:
                best_cost, best = cost, i
        if best is not None:
            f1, x1, y1, tid, team = ends.pop(best)
            union(int(r.track_id), int(tid))
        ends.append((r.f1, r.x1, r.y1, int(r.track_id), int(r.team)))
        # prune long-dead entries
        ends = [e for e in ends if r.f0 - e[0] <= max_gap_f]

    # Pass 2 — meta-level, across long invisible stretches (close-up periods
    # end every track and can outlast the kinematic horizon). Rationale: when
    # the wide camera returns, players are usually near where it left them,
    # so allow long gaps but demand SHORT distance. Conservative by design.
    for _ in range(2):
        groups = {}
        for t in parent:
            groups.setdefault(find(t), []).append(t)
        idx = s.set_index("track_id")
        metas = []
        for mid, tids in groups.items():
            rows = idx.loc[tids]
            first = rows.sort_values("f0").iloc[0]
            last = rows.sort_values("f1").iloc[-1]
            metas.append(dict(mid=mid, team=int(rows["team"].mode().iat[0]),
                              f0=int(first.f0), x0=first.x0, y0=first.y0,
                              f1=int(last.f1), x1=last.x1, y1=last.y1))
        metas.sort(key=lambda m: m["f0"])
        ends2 = []
        merged_any = False
        for m in metas:
            best, best_cost = None, np.inf
            for i, e in enumerate(ends2):
                gap = m["f0"] - e["f1"]
                if gap <= 0 or gap > int(60.0 * fps):
                    continue
                if e["team"] != m["team"]:
                    continue
                d = float(np.hypot(m["x0"] - e["x1"], m["y0"] - e["y1"]))
                if d > 14.0:
                    continue
                cost = d + gap / fps * 0.05
                if cost < best_cost:
                    best_cost, best = cost, i
            if best is not None:
                e = ends2.pop(best)
                union(m["mid"], e["mid"])
                merged_any = True
                e2 = dict(e)
                e2.update(f1=m["f1"], x1=m["x1"], y1=m["y1"])
                ends2.append(e2)
            else:
                ends2.append(m)
        if not merged_any:
            break

    out = s[["track_id", "team", "n"]].copy()
    out["meta_id"] = [find(int(t)) for t in out["track_id"]]
    out = out.rename(columns={"n": "frames"})
    return out.sort_values(["meta_id", "track_id"]).reset_index(drop=True)


def meta_map(gs) -> dict:
    """{track_id: meta_id} from conservative consolidation — lets exports
    aggregate stats per (approximate) player before any naming happens."""
    m = consolidate_tracks(gs)
    return dict(zip(m.track_id.astype(int), m.meta_id.astype(int)))


def consolidation_report(meta: pd.DataFrame) -> str:
    per = meta.groupby("meta_id").agg(tracks=("track_id", "size"),
                                      frames=("frames", "sum"),
                                      team=("team", "first"))
    per = per.sort_values("frames", ascending=False)
    total = per["frames"].sum()
    top = per.head(40)
    cover = top["frames"].sum() / max(total, 1)
    lines = [
        f"tracks: {len(meta)} -> meta-tracks: {len(per)}",
        f"top-40 meta-tracks cover {cover*100:.0f}% of player-frames "
        f"(a full lineup is 22 starters + subs + officials)",
        f"largest: " + ", ".join(
            f"m{mid}({int(r.tracks)}trk/{int(r.frames)}f/t{int(r.team)})"
            for mid, r in per.head(8).iterrows()),
    ]
    return "\n".join(lines)


# ── Identity file ────────────────────────────────────────────────────────────

def identity_path(slug: str, period: int = 1) -> Path:
    """Per-period identity file — track ids restart every pipeline run, so
    each half needs its own naming. Naming the same player identically in
    both halves unifies them across the match (exports and the report
    aggregate by name)."""
    return identities_dir() / f"{slug}_p{int(period)}.json"


def save_identities(slug: str, meta: pd.DataFrame, names: dict,
                    period: int = 1) -> Path:
    """``names``: {meta_id: {"name": str, "number": int|None}}. Persists both
    the naming and the track->meta mapping so the export can resolve tracks."""
    payload = {
        "slug": slug,
        "period": int(period),
        "meta_of_track": {str(int(r.track_id)): int(r.meta_id)
                          for r in meta.itertuples(index=False)},
        "players": {str(int(k)): v for k, v in names.items()},
    }
    p = identity_path(slug, period)
    p.write_text(json.dumps(payload, indent=2))
    return p


def load_identity_map(slug: str, period: int = 1) -> Optional[dict]:
    """{track_id: {"name":…, "number":…}} for ``period``'s track ids, or None.
    Falls back to the legacy single file ``{slug}.json`` when its period
    matches (files without a period field predate per-half artifacts)."""
    p = identity_path(slug, period)
    if not p.exists():
        legacy = identities_dir() / f"{slug}.json"
        if not legacy.exists():
            return None
        d = json.loads(legacy.read_text())
        if int(d.get("period", 1)) != int(period):
            return None
    else:
        d = json.loads(p.read_text())
    meta_of = {int(k): int(v) for k, v in d["meta_of_track"].items()}
    players = {int(k): v for k, v in d["players"].items()}
    return {tid: players[mid] for tid, mid in meta_of.items()
            if mid in players}


# ── Naming widget (notebook) ─────────────────────────────────────────────────

def build_identity_widget(gs, meta: pd.DataFrame, top_n: int = 40,
                          crops_per_meta: int = 3):
    """ipywidgets gallery: best crops per meta-track + name/number inputs.
    Returns the widget; 'Save' writes data/identities/{slug}.json."""
    import cv2
    import ipywidgets as W
    from IPython.display import display

    video = str(Config.MATCH_VIDEOS[gs.slug])
    pl = gs.players
    per = meta.groupby("meta_id").agg(frames=("frames", "sum"),
                                      team=("team", "first"))
    order = per.sort_values("frames", ascending=False).head(top_n).index

    cap = cv2.VideoCapture(video)
    rows, inputs = [], {}
    for mid in order:
        tids = meta.loc[meta.meta_id == mid, "track_id"]
        sub = pl[pl.track_id.isin(tids)].sort_values("frame")
        # biggest boxes = closest to camera = most readable crops
        sub = sub.assign(h=sub.y2 - sub.y1).nlargest(60, "h")
        picks = sub.iloc[:: max(len(sub) // crops_per_meta, 1)][:crops_per_meta]
        imgs = []
        for r in picks.itertuples(index=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
            ok, img = cap.read()
            if not ok:
                continue
            pad = 10
            crop = img[max(int(r.y1) - pad, 0):int(r.y2) + pad,
                       max(int(r.x1) - pad, 0):int(r.x2) + pad]
            if crop.size == 0:
                continue
            ok2, buf = cv2.imencode(".jpg", crop)
            if ok2:
                imgs.append(W.Image(value=buf.tobytes(), format="jpg",
                                    layout=W.Layout(height="120px")))
        name = W.Text(placeholder="name", layout=W.Layout(width="160px"))
        num = W.Text(placeholder="#", layout=W.Layout(width="50px"))
        inputs[int(mid)] = (name, num)
        label = W.HTML(f"<b>m{mid}</b> team{int(per.loc[mid,'team'])} "
                       f"{int(per.loc[mid,'frames'])}f")
        rows.append(W.HBox([label, *imgs, num, name]))
    cap.release()

    status = W.HTML()
    btn = W.Button(description="Save identities", button_style="success")

    def _save(_):
        names = {}
        for mid, (name, num) in inputs.items():
            if name.value.strip() or num.value.strip():
                names[mid] = {"name": name.value.strip() or None,
                              "number": int(num.value) if num.value.strip().isdigit() else None}
        p = save_identities(gs.slug, meta, names,
                            period=int(gs.meta.get("period", 1)))
        status.value = f"saved {len(names)} players -> {p}"

    btn.on_click(_save)
    box = W.VBox([*rows, W.HBox([btn, status])])
    display(box)
    return box


def main():
    ap = argparse.ArgumentParser(description="Consolidate tracks into meta-tracks")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--half", type=int, default=None, choices=[1, 2],
                    help="period to consolidate (default: the only stored one)")
    args = ap.parse_args()
    from .game_state import GameState
    gs = GameState.load(args.match, period=args.half)
    meta = consolidate_tracks(gs)
    print(consolidation_report(meta))


if __name__ == "__main__":
    main()
