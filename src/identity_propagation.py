"""
Identity propagation: spread identity anchors — human review labels AND
gate-passed automatic jersey numbers — to the hundreds of unlabelled track
fragments, so event attribution rises toward usable coverage without more
human time. On matches with no human identity file the jersey seeds are the
only anchors, which makes this the fully-automatic identity path.

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
import os
from collections import defaultdict
from typing import Optional

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

# ReID cross-cut link tunables (Phase 3). Chosen on the END-TO-END pooled
# SofaScore player-tier metric across all 7 truth matches (124 players), which
# is the real product signal — not the held-out k-fold precision (that measures
# only the hard anchor-vs-anchor case and understates coverage's value):
#   baseline (no ReID) : rho 0.360  recall 16.6%  attribution 14.8%
#   0.94/topk1         : rho 0.368  recall 19.7%  attribution 17.6%
#   0.93/topk2         : rho 0.363  recall 27.5%  attribution 24.6%
#   0.92/topk3         : rho 0.433  recall 43.1%  attribution 38.2%  <- chosen
# The aggressive point wins on ALL THREE metrics: extra coverage more than
# offsets the lower per-link precision (0.83 held-out) — pooled rho actually
# RISES because more of each player's passes are captured. Confirmed a stable
# optimum, not a spike: 0.91/topk3 gives rho 0.431 / recall 43.2% (≈identical),
# so topk=3 is the driver. Team-level is untouched (ReID links are same-team
# only, so a mis-link moves a pass between teammates, never across teams).
# Overridable via env for future sweeps.
REID_LINK_MIN = float(os.environ.get("REID_LINK_MIN", "0.92"))
REID_TOPK = int(os.environ.get("REID_TOPK", "3"))
# Uniqueness-demoted jersey metas seed propagation only above this vote count
# (the KEPT meta of each (team,number) always seeds). Marginal 3-vote demoted
# fragments are where residual misreads hide; the floor is swept end-to-end
# against the pooled SofaScore player tier.
DEMOTED_SEED_VOTE_MIN = int(os.environ.get("DEMOTED_SEED_VOTE_MIN", "3"))


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

    def __init__(self, gs, slug: str, period: int, fps: float | None = None,
                 reid_embs: dict | None = None,
                 reid_link_min: float = REID_LINK_MIN,
                 reid_topk: int = REID_TOPK):
        self.fps = fps or gs.fps
        ep = _endpoints(gs)
        self.tracks = [int(t) for t in ep.index]
        self.team = {int(t): int(v) for t, v in ep["team"].items()}
        self.reads = _jersey_reads(slug, period)
        self.slug, self.period = slug, period
        self._build_links(ep)
        self.n_reid_links = 0
        if reid_embs:
            self._build_reid_links(ep, reid_embs, reid_link_min, reid_topk)

    def _build_reid_links(self, ep, embs: dict, link_min: float, topk: int):
        """Add cross-cut appearance links to :attr:`link`. This is the piece
        kinematic handoff structurally cannot do: a player's fragments on
        opposite sides of a camera cut have no position continuity, but a
        purpose-trained ReID embedding (osnet_ain, P(same>cross)=0.84 measured)
        still recognises them. Precision guards, in order of strength:
        (1) SAME TEAM only; (2) NO TEMPORAL OVERLAP — two tracks alive in the
        same frame are on-screen simultaneously, so they are definitely
        different players and can never link; (3) MUTUAL top-k nearest above a
        cosine floor (the discipline that made kinematic handoff precision
        1.00); (4) JERSEY CONFLICT VETO — if both fragments OCR'd a shirt number
        and the pluralities differ, they are different players. The unanimity
        veto in :meth:`spread` is the final backstop when a track's neighbours
        disagree."""
        f0 = ep["f0"].to_dict(); f1 = ep["f1"].to_dict()
        tracks = [t for t in self.tracks if t in embs]
        by_team: dict = defaultdict(list)
        for t in tracks:
            by_team[self.team[t]].append(t)
        added = 0
        for ts in by_team.values():
            n = len(ts)
            if n < 2:
                continue
            M = np.stack([embs[t] for t in ts]).astype(np.float32)  # L2-normed
            S = M @ M.T
            for i in range(n):
                S[i, i] = -1.0
                ai = ts[i]
                for j in range(n):
                    if i == j:
                        continue
                    bj = ts[j]
                    overlap = not (f1[ai] < f0[bj] or f1[bj] < f0[ai])
                    if overlap:
                        S[i, j] = -1.0
            cand = [set() for _ in range(n)]
            for i in range(n):
                order = np.argsort(-S[i])
                for j in order[:topk]:
                    if S[i, j] >= link_min:
                        cand[i].add(int(j))
            for i in range(n):
                for j in cand[i]:
                    if i not in cand[j] or j <= i:      # mutual, once
                        continue
                    a, b = ts[i], ts[j]
                    ra, rb = self.reads.get(a), self.reads.get(b)
                    if ra and rb and ra[0] != rb[0]:    # jersey conflict veto
                        continue
                    self.link.setdefault(a, set()).add(b)
                    self.link.setdefault(b, set()).add(a)
                    added += 1
        self.n_reid_links = added

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


def _load_reid(slug: str, period: int) -> dict | None:
    """Cached per-track ReID embeddings (osnet_ain) if built, else None."""
    try:
        from . import reid as reid_mod
        return reid_mod.load_embeddings(slug, period)
    except Exception:
        return None


def propagate(gs, anchors: dict, slug: str, period: int,
              fps: float | None = None, verbose: bool = False,
              reid_embs: dict | None = None) -> dict:
    if reid_embs is None:
        reid_embs = _load_reid(slug, period)
    g = HandoffGraph(gs, slug, period, fps=fps, reid_embs=reid_embs)
    labels = g.spread(anchors)
    if verbose:
        print(f"[{slug} p{period}] {len(anchors)} anchors -> {len(labels)} "
              f"labelled ({len(g.reads)} jersey reads, "
              f"{g.n_reid_links} ReID links)")
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


def _jersey_meta_anchors(gs, slug: str, period: int) -> tuple:
    """Anchors synthesized from gate-passed confident jersey metas — the
    AUTOMATIC seed set. Every member track of a confident meta (structural
    gates: corroboration / conflict veto / single-support floor; measured
    zero-wrong on 13 GT tracks) is seeded with ("num", team, number), exactly
    the label the export would give those tracks anyway via
    ``resolve_player``'s jersey path — propagation then spreads it to linked
    fragments the meta itself never reached. Returns (anchors, key_info)."""
    from .jersey_ocr import load_jersey_numbers, jersey_path
    from .identity import meta_map as build_meta_map
    # include_demoted: same-(team,number) metas all seed the SAME key — under
    # a DENSE, per-crop-accurate reader (the VLM) they are the fragmented
    # player himself, so seeding them adds coverage without ambiguity.
    # MEASURED to reverse for easyocr (sut-mla pooled rho 0.42 -> 0.04):
    # sparse reads with structured digit confusions (6<->8, 16<->18) let a
    # consistently-misread demoted meta pass the gates and spread a wrong key
    # over a real player's fragments — so easyocr files keep the conservative
    # uniqueness-filtered set.
    reader = ""
    jp = jersey_path(slug, period)
    if jp.exists():
        reader = json.loads(jp.read_text(encoding="utf-8")) \
            .get("params", {}).get("reader", "easyocr")
    jn = load_jersey_numbers(slug, period,
                             include_demoted=(reader == "qwen2-vl"))
    if not jn:
        return {}, {}
    members = defaultdict(list)
    for tid, mid in build_meta_map(gs).items():
        members[int(mid)].append(int(tid))
    anchors, key_info = {}, {}
    for mid, rec in jn.items():
        team, num = rec.get("team"), rec.get("number")
        if team not in (0, 1) or num is None:
            continue
        if ("demoted_for" in rec
                and int(rec.get("votes", 0)) < DEMOTED_SEED_VOTE_MIN):
            continue
        key = ("num", int(team), int(num))
        key_info.setdefault(key, {"number": int(num), "team": int(team)})
        for tid in members.get(int(mid), [int(mid)]):
            anchors[int(tid)] = key
    return anchors, key_info


def expanded_identity_map(slug: str, period: int) -> Optional[dict]:
    """{track_id: {name, number, team}} — identity seeds EXPANDED by validated
    propagation (mutual-best kinematics + ReID cross-cut links + jersey).

    Seeds, in priority order: the human identity file (exact entries always
    kept, their info defines the label for a key) UNIONed with automatic
    confident-jersey-meta anchors (:func:`_jersey_meta_anchors`). The union
    matters in both regimes: on human-labeled matches jersey metas add keys
    the human never labeled; on unlabeled matches (the production-cost path)
    they are the ONLY seed and this function previously returned None —
    leaving propagation entirely unused exactly where automation matters most.
    Returns None only when there are no seeds of either kind."""
    from .game_state import GameState
    idmap = load_identity_map(slug, period)
    gs = GameState.load(slug, period=period)
    anchors, key_info = _jersey_meta_anchors(gs, slug, period)
    if idmap:
        human_keys = set()
        for tid, info in idmap.items():
            k = _anchor_key(info)
            if k is not None:
                anchors[int(tid)] = k          # human beats jersey per track
                if k not in human_keys:        # first human anchor defines it
                    key_info[k] = info
                    human_keys.add(k)
    if not anchors:
        return dict(idmap) if idmap else None
    labels = propagate(gs, anchors, slug, period)
    out = dict(idmap) if idmap else {}         # keep exact human entries
    for tid, k in labels.items():
        if tid not in out and k in key_info:
            out[int(tid)] = key_info[k]
    return out


def _kfold(g: "HandoffGraph", anchors: dict, folds: int = 5,
          seed: int = 0) -> dict:
    """Hold out 1/folds of anchors, spread from the rest, score on the held-out
    tracks the spread reached. Precision = correct/reached; recall = reached/held."""
    rng = np.random.default_rng(seed)
    tids = np.array(sorted(anchors))
    rng.shuffle(tids)
    fold_of = {int(t): i % folds for i, t in enumerate(tids)}
    tp = fp = labelled = held = 0
    for f in range(folds):
        train = {t: k for t, k in anchors.items() if fold_of[t] != f}
        test = {t: k for t, k in anchors.items() if fold_of[t] == f}
        got = g.spread(train)
        for t, truth in test.items():
            held += 1
            if t in got:
                labelled += 1
                tp += int(got[t] == truth)
                fp += int(got[t] != truth)
    prec = tp / labelled if labelled else float("nan")
    rec = labelled / held if held else float("nan")
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp,
            "labelled": labelled, "held": held}


def evaluate(slug: str, period: int, folds: int = 5, seed: int = 0) -> dict:
    """K-fold precision/recall of kinematic+jersey propagation (no ReID)."""
    gs = GameState.load(slug, period=period)
    anchors = _anchor_map_from_identity(gs, slug, period)
    if len(anchors) < folds:
        print(f"[{slug} p{period}] only {len(anchors)} anchors - skip eval")
        return {}
    g = HandoffGraph(gs, slug, period)
    r = _kfold(g, anchors, folds, seed)
    print(f"[{slug} p{period}] propagation CV: precision {r['precision']:.2f} "
          f"({r['tp']}/{r['labelled']} correct), recall {r['recall']:.2f} "
          f"({r['labelled']}/{r['held']} held-out reached), "
          f"{len(anchors)} anchors")
    return {**r, "anchors": len(anchors)}


def evaluate_reid(slug: str, period: int, folds: int = 5, seed: int = 0,
                  link_min: float = REID_LINK_MIN,
                  topk: int = REID_TOPK) -> dict:
    """Isolate the ReID contribution: same held-out k-fold, WITHOUT vs WITH the
    cross-cut ReID links. Uses anchor-track embeddings only (fast — reuses the
    discriminative-test crop path), which is exactly what precision needs since
    the held-out tracks are anchors. A win = recall up, precision held."""
    from . import reid as reid_mod
    gs = GameState.load(slug, period=period)
    anchors = _anchor_map_from_identity(gs, slug, period)
    if len(anchors) < folds:
        print(f"[{slug} p{period}] only {len(anchors)} anchors - skip")
        return {}
    embs = reid_mod.embed_tracks(slug, period, list(anchors), gs=gs)
    g0 = HandoffGraph(gs, slug, period)                       # kinematic only
    g1 = HandoffGraph(gs, slug, period, reid_embs=embs,
                      reid_link_min=link_min, reid_topk=topk)  # + ReID
    r0 = _kfold(g0, anchors, folds, seed)
    r1 = _kfold(g1, anchors, folds, seed)
    print(f"[{slug} p{period}] anchors={len(anchors)} "
          f"reid_links={g1.n_reid_links} (min={link_min}, topk={topk})")
    print(f"    kinematic : P {r0['precision']:.2f}  R {r0['recall']:.2f}  "
          f"({r0['tp']}/{r0['labelled']} correct, {r0['labelled']}/{r0['held']} reached)")
    print(f"    + ReID    : P {r1['precision']:.2f}  R {r1['recall']:.2f}  "
          f"({r1['tp']}/{r1['labelled']} correct, {r1['labelled']}/{r1['held']} reached)")
    return {"kinematic": r0, "reid": r1, "anchors": len(anchors),
            "reid_links": g1.n_reid_links}


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
    ap.add_argument("--reid", action="store_true",
                    help="k-fold WITHOUT vs WITH cross-cut ReID links")
    ap.add_argument("--link_min", type=float, default=REID_LINK_MIN)
    ap.add_argument("--topk", type=int, default=REID_TOPK)
    args = ap.parse_args()
    periods = [args.half] if args.half else available_periods(args.match)
    for p in periods:
        if args.reid:
            evaluate_reid(args.match, p, link_min=args.link_min, topk=args.topk)
        elif args.eval:
            evaluate(args.match, p)
            coverage_gain(args.match, p)
        else:
            coverage_gain(args.match, p)


if __name__ == "__main__":
    main()
