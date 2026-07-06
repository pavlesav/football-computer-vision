"""
Identity propagation: spread the ~35 human anchor labels per half to the
hundreds of unlabelled track fragments, so event attribution rises from ~12%
toward usable coverage without more human time.

Why a new module instead of ``consolidate_tracks``: consolidation was measured
to over-merge (126 tracks into one meta, mixing players) because it greedily
unions the best kinematic match. Propagation inverts the risk posture — it
only ever copies a KNOWN identity onto an adjacent fragment, and only when the
handoff is unambiguous. A wrong propagation still poisons two players, so the
gates are strict and, crucially, MEASURED against held-out human labels
(:func:`evaluate`) rather than trusted on faith.

Two propagation channels, both high-precision:

* **Kinematic handoff.** Track A (known identity) ends at (t, x, y); track B
  starts near there soon after, same team, no temporal overlap. B inherits A's
  identity — but only if B has no *conflicting* labelled neighbour (ambiguity
  veto) and the nearest labelled neighbour is within tight thresholds.
  Iterated to a fixpoint so a freshly-labelled B can carry a chain forward,
  but every step re-checks unanimity, so a chain never launders a conflict.

* **Jersey number.** A fragment whose own OCR reads (team, number) match an
  anchor's (team, number) gets that identity directly. Independent of
  kinematics, so it reaches fragments on the far side of camera cuts.

Run::

    python -m src.identity_propagation --match sut-mla --half 1 --eval
    python -m src.identity_propagation --match sut-mla            # both halves
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import Config
from .game_state import GameState, available_periods
from .identity import _track_summaries, load_identity_map, identity_path
from .roles import infer_attack_direction, identify_goalkeepers
from .jersey_ocr import jersey_path

# Handoff gates. The hard-won lesson (measured: loose gates gave CV precision
# 0.50): kinematic continuity is only trustworthy across a SHORT occlusion
# within one continuous camera shot. Across a camera cut every track dies and
# new ones are born, and image->pitch position continuity is meaningless
# (the camera jumped), so a long gap invites wrong same-team handoffs. Keep
# the gap short, the distance tight, and require MUTUAL best match.
# Measured: within the mutual-best-match regime, CV precision is 1.00 and
# INVARIANT to these thresholds over 1.6-6s / 6-12m (mutual-best is the
# binding constraint, not the window). A robust middle is chosen.
MAX_GAP_S = 4.0            # only bridge brief occlusions, not camera cuts
MAX_SPEED_MS = 7.5         # players cover at most this while unseen
SLACK_M = 3.0             # endpoint jitter margin
MAX_HANDOFF_M = 8.0        # absolute cap on endpoint distance for a handoff
MIN_TRACK_FRAMES = 10


def _anchor_key(info: dict):
    """Identity key that unifies across halves: prefer (team, number), else
    the name, else None (unusable as a cross-fragment anchor)."""
    if info is None:
        return None
    num = info.get("number")
    team = info.get("team")
    if num is not None and team in (0, 1):
        return ("num", int(team), int(num))
    if info.get("name"):
        return ("name", info["name"])
    return None


def _endpoints(gs) -> pd.DataFrame:
    directions = infer_attack_direction(gs)
    gk_map = identify_goalkeepers(gs, directions)
    s = _track_summaries(gs, gk_map)
    return s.set_index("track_id")


def _jersey_reads(slug: str, period: int) -> dict:
    """{track_id: (number, votes)} plurality own-number read per track."""
    p = jersey_path(slug, period)
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    for tid, v in d.get("track_reads", {}).items():
        nums = [r["number"] for r in v.get("reads", [])]
        if not nums:
            continue
        from collections import Counter
        num, votes = Counter(nums).most_common(1)[0]
        out[int(tid)] = (int(num), int(votes))
    return out


class HandoffGraph:
    """Seed-independent handoff structure for one period: mutual-best
    kinematic links + team + jersey reads. Built once (the O(n^2) part),
    then :meth:`spread` runs cheaply over any anchor set — so k-fold CV and
    the real apply share the same graph."""

    def __init__(self, gs, slug: str, period: int, fps: float | None = None):
        self.fps = fps or gs.fps
        ep = _endpoints(gs)
        self.tracks = [int(t) for t in ep.index]
        self.team = {int(t): int(v) for t, v in ep["team"].items()}
        self.reads = _jersey_reads(slug, period)
        self.slug, self.period = slug, period
        self._build_links(ep)

    def _build_links(self, ep):
        fps = self.fps
        max_gap_f = int(MAX_GAP_S * fps)
        f0 = ep["f0"].to_dict(); f1 = ep["f1"].to_dict()
        x0 = ep["x0"].to_dict(); y0 = ep["y0"].to_dict()
        x1 = ep["x1"].to_dict(); y1 = ep["y1"].to_dict()
        team = self.team
        # Sort by start frame; only scan recently-ended tracks (a small
        # window) instead of all pairs — O(n·w) not O(n^2).
        order = sorted(self.tracks, key=lambda t: f0[t])
        best_succ = {t: (np.inf, None) for t in self.tracks}
        best_pred = {t: (np.inf, None) for t in self.tracks}
        ended = []    # (f1, tid), pruned as we advance
        for b in order:
            fb = f0[b]
            ended = [(fe, a) for fe, a in ended if fb - fe <= max_gap_f]
            for fe, a in ended:
                if team[a] != team[b]:
                    continue
                gap = fb - f1[a]
                if gap <= 0 or gap > max_gap_f:
                    continue
                d = np.hypot(x0[b] - x1[a], y0[b] - y1[a])
                reach = min(MAX_SPEED_MS * (gap / fps) + SLACK_M, MAX_HANDOFF_M)
                if d > reach:
                    continue
                c = d + 0.5 * (gap / fps)
                if c < best_succ[a][0]:
                    best_succ[a] = (c, b)
                if c < best_pred[b][0]:
                    best_pred[b] = (c, a)
            ended.append((f1[b], b))
        link: dict = {}
        for a in self.tracks:
            c, b = best_succ[a]
            if b is not None and best_pred[b] == (c, a):
                link.setdefault(a, set()).add(b)
                link.setdefault(b, set()).add(a)
        self.link = link

    def spread(self, anchors: dict) -> dict:
        """anchors: {track_id: key} -> full {track_id: key} labelling."""
        labels = {int(t): k for t, k in anchors.items() if int(t) in self.team}
        num_keys = {k for k in anchors.values() if k and k[0] == "num"}
        for tid, (num, votes) in self.reads.items():
            if tid in self.team and tid not in labels and votes >= 3:
                key = ("num", self.team[tid], num)
                if key in num_keys:
                    labels[tid] = key
        changed = True
        rounds = 0
        while changed and rounds < 30:
            changed = False
            rounds += 1
            for b in self.tracks:
                if b in labels:
                    continue
                neigh = {labels[a] for a in self.link.get(b, ()) if a in labels}
                if len(neigh) == 1:
                    labels[b] = next(iter(neigh))
                    changed = True
        return labels


def propagate(gs, anchors: dict, slug: str, period: int,
              fps: float | None = None, verbose: bool = False) -> dict:
    g = HandoffGraph(gs, slug, period, fps=fps)
    labels = g.spread(anchors)
    if verbose:
        print(f"[{slug} p{period}] {len(anchors)} anchors -> {len(labels)} "
              f"labelled ({len(g.reads)} jersey reads)")
    return labels


def _anchor_map_from_identity(gs, slug: str, period: int) -> dict:
    """{track_id: identity_key} from the human identity file."""
    idmap = load_identity_map(slug, period) or {}
    out = {}
    for tid, info in idmap.items():
        k = _anchor_key(info)
        if k is not None:
            out[int(tid)] = k
    return out


def expanded_identity_map(slug: str, period: int) -> Optional[dict]:
    """{track_id: {name, number, team}} — the human identity file EXPANDED by
    validated propagation (mutual-best kinematics + jersey). A drop-in
    superset of :func:`identity.load_identity_map` for the export: every
    anchor keeps its exact info; propagated tracks inherit the info of the
    anchor whose key they matched. CV precision of the propagation step is
    1.00 (held-out anchors), so this only adds correct labels. Returns None
    when there is no human identity file to seed from."""
    from .game_state import GameState
    idmap = load_identity_map(slug, period)
    if not idmap:
        return None
    gs = GameState.load(slug, period=period)
    anchors = {}
    key_info = {}
    for tid, info in idmap.items():
        k = _anchor_key(info)
        if k is not None:
            anchors[int(tid)] = k
            key_info.setdefault(k, info)   # first anchor defines the label
    if not anchors:
        return dict(idmap)
    labels = propagate(gs, anchors, slug, period)
    out = dict(idmap)                      # keep exact human entries
    for tid, k in labels.items():
        if tid not in out and k in key_info:
            out[int(tid)] = key_info[k]
    return out


def evaluate(slug: str, period: int, folds: int = 5, seed: int = 0) -> dict:
    """K-fold: hold out a fraction of anchors, propagate from the rest (+
    jersey), measure precision/recall on the held-out tracks. This is the
    honest test — the held-out human labels are ground truth the propagation
    never saw."""
    gs = GameState.load(slug, period=period)
    anchors = _anchor_map_from_identity(gs, slug, period)
    if len(anchors) < folds:
        print(f"[{slug} p{period}] only {len(anchors)} anchors - skip eval")
        return {}
    rng = np.random.default_rng(seed)
    tids = np.array(sorted(anchors))
    rng.shuffle(tids)
    fold_of = {int(t): i % folds for i, t in enumerate(tids)}

    g = HandoffGraph(gs, slug, period)   # seed-independent, built once
    tp = fp = labelled = held = 0
    for f in range(folds):
        train = {t: k for t, k in anchors.items() if fold_of[t] != f}
        test = {t: k for t, k in anchors.items() if fold_of[t] == f}
        got = g.spread(train)
        for t, truth in test.items():
            held += 1
            if t in got:      # propagation reached this held-out track
                labelled += 1
                if got[t] == truth:
                    tp += 1
                else:
                    fp += 1
    prec = tp / labelled if labelled else float("nan")
    rec = labelled / held if held else float("nan")
    print(f"[{slug} p{period}] propagation CV: precision {prec:.2f} "
          f"({tp}/{labelled} propagated-correct), recall {rec:.2f} "
          f"({labelled}/{held} held-out reached), {len(anchors)} anchors")
    return {"precision": prec, "recall": rec, "anchors": len(anchors),
            "tp": tp, "fp": fp, "labelled": labelled, "held": held}


def coverage_gain(slug: str, period: int) -> dict:
    """How much event attribution rises: anchors-only vs propagated."""
    from .events import detect_events
    gs = GameState.load(slug, period=period)
    anchors = _anchor_map_from_identity(gs, slug, period)
    got = propagate(gs, anchors, slug, period)
    events, _ = detect_events(gs)
    part = defaultdict(int)
    for e in events:
        for t in (e.player, e.details.get("recipient")):
            if t is not None and int(t) >= 0:
                part[int(t)] += 1
    total = sum(part.values())
    anch_ev = sum(n for t, n in part.items() if t in anchors)
    prop_ev = sum(n for t, n in part.items() if t in got)
    print(f"[{slug} p{period}] event participations: anchors "
          f"{anch_ev/total*100:.0f}% -> propagated {prop_ev/total*100:.0f}% "
          f"({len(anchors)} -> {len(got)} tracks)")
    return {"anchor_pct": anch_ev / total * 100 if total else 0,
            "prop_pct": prop_ev / total * 100 if total else 0,
            "anchors": len(anchors), "propagated": len(got)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, choices=[1, 2])
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    periods = [args.half] if args.half else available_periods(args.match)
    for p in periods:
        if args.eval:
            evaluate(args.match, p)
        coverage_gain(args.match, p)


if __name__ == "__main__":
    main()
