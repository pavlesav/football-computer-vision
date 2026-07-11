"""
Unsupervised team assignment from OSNet ReID embeddings (SoccerNet-GSR SOTA
pattern) — validated against our human-labeled team classifier.

Why: the production team classifier is supervised per match (~15 min human
labeling each) and is the diagnosed weakest link on night games (mla-bud-2
player rho 0.07; kit colors wash out under floodlights for the ResNet18+KNN
trained on masked crops). Both the SoccerNet-2025 GSR winner and the previous
SOTA ("From Broadcast to Minimap", CVPRW 2025) assign teams by *clustering
appearance-ReID embeddings* — no supervision — into team/referee/GK groups and
assigning by cosine distance to cluster centers. We already cache exactly those
embeddings per half (``reid.npz``, osnet_ain_x1_0, see :mod:`src.reid`), so the
hypothesis is directly testable against 14 halves of human-labeled ground
truth before any production wiring.

Design:
* Cluster per-track mean embeddings (L2-normed, so k-means == cosine k-means)
  with k clusters (default 4: two teams + refs + GK-ish). The two most
  populous clusters are the outfield teams — refs and GKs are numerically
  rare and split off first because their kit differs most.
* ``--eval``: for every half with reid.npz, map the two team clusters onto
  classifier team ids by majority overlap (2x2 Hungarian = argmax of the two
  pairings) and report agreement on tracks the classifier put in a team.
  High agreement on day games validates the signal; disagreement pockets on
  night games are CANDIDATE CLASSIFIER ERRORS, not necessarily ours.
* ``--apply``: conservative repair in the :mod:`src.team_repair` spirit —
  only flip a track's stored team when the clustering is CONFIDENT it sits
  with the other team (margin gate on cosine distance to the two team
  centroids) and the track is not role-identified as a GK. Writes a backup of
  players.parquet next to it; measured end-to-end via sofa_eval before/after.

Run::

    python -m src.team_cluster --eval                 # all halves, agreement
    python -m src.team_cluster --match mla-bud-2 --half 1 --apply
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter

import numpy as np
import pandas as pd

from .config import Config
from .game_state import GameState, available_periods, game_state_dir
from .reid import load_embeddings

K_CLUSTERS = 4
# --apply flips a track only when the winning team centroid is closer by this
# cosine margin AND the cluster-vs-stored disagreement is unambiguous.
APPLY_MARGIN = 0.05
MIN_TRACK_ROWS = 10


def _track_teams(gs: GameState) -> dict:
    """{track_id: stored team (row-majority)} for substantial tracks."""
    pl = gs.players
    counts = pl.groupby("track_id").size()
    keep = counts[counts >= MIN_TRACK_ROWS].index
    sub = pl[pl.track_id.isin(keep)]
    return {int(t): int(m) for t, m in
            sub.groupby("track_id")["team_id"].agg(lambda s: s.mode().iat[0])
            .items()}


def _kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0):
    """Plain cosine k-means (X is L2-normed): returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    # k-means++ style init on cosine distance
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = 1.0 - np.max(np.stack([X @ c for c in centers], 1), axis=1)
        d = np.clip(d, 1e-9, None) ** 2
        centers.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.stack(centers)
    for _ in range(iters):
        sim = X @ C.T
        lab = sim.argmax(1)
        newC = np.stack([
            X[lab == j].mean(0) if np.any(lab == j) else C[j]
            for j in range(k)])
        newC /= (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-8)
        if np.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return (X @ C.T).argmax(1), C


# Grouping thresholds, validated across the 14 labeled halves:
# SEED_DEDUP — a candidate team-B seed above this cosine to seed A is a
#   sub-cluster of the SAME kit (sut-pet p2: one team split 157+146, mutually
#   ~0.9+; naive 'two biggest = two teams' paired those two fragments).
# JOIN_MIN — a leftover cluster below this to both seeds is refs/other
#   (mla-bud-2 refs sit at 0.75-0.78 to the teams; night-game team-vs-team
#   similarity reaches 0.79-0.94, which is also why plain agglomerative
#   merging collapsed the two teams into one group there).
SEED_DEDUP = 0.90
JOIN_MIN = 0.80


def _merge_to_groups(C: np.ndarray, sizes: list) -> list:
    """Group k-means clusters into team A (0), team B (1), other (2) —
    fully unsupervised, seed-based. Returns group id per cluster."""
    order = sorted(range(len(C)), key=lambda j: -sizes[j])
    seed_a = order[0]
    seed_b = None
    for j in order[1:]:
        if float(C[j] @ C[seed_a]) < SEED_DEDUP:
            seed_b = j
            break
    if seed_b is None:                      # degenerate: everything one kit
        seed_b = order[1] if len(order) > 1 else order[0]
    out = [2] * len(C)
    out[seed_a], out[seed_b] = 0, 1
    for j in range(len(C)):
        if j in (seed_a, seed_b):
            continue
        sa, sb = float(C[j] @ C[seed_a]), float(C[j] @ C[seed_b])
        if max(sa, sb) < JOIN_MIN:
            continue                        # refs / outliers stay 'other'
        out[j] = 0 if sa >= sb else 1
    return out


def cluster_half(slug: str, period: int, k: int = K_CLUSTERS,
                 verbose: bool = True) -> dict:
    """Cluster a half's cached ReID embeddings, merge clusters into 3 groups
    (two teams + other/refs, fully unsupervised), then compare the two
    largest groups against stored classifier teams. Returns per-track
    assignments + agreement stats (or {} when inputs are missing)."""
    embs = load_embeddings(slug, period)
    if not embs:
        if verbose:
            print(f"[{slug} p{period}] no reid.npz — skip "
                  f"(python -m src.reid --match {slug} --half {period})")
        return {}
    gs = GameState.load(slug, period=period)
    stored = _track_teams(gs)
    tids = [t for t in sorted(embs) if t in stored]
    if len(tids) < 40:
        if verbose:
            print(f"[{slug} p{period}] only {len(tids)} labeled embedded "
                  f"tracks — skip")
        return {}
    X = np.stack([embs[t] for t in tids]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab, C = _kmeans(X, k)

    sizes = [int(np.sum(lab == j)) for j in range(k)]
    group_of = _merge_to_groups(C, sizes)
    team_groups = [0, 1]
    a_g, b_g = team_groups

    # group centroids (size-weighted) for margins / nearest-team
    def group_centroid(g):
        idx = [j for j in range(k) if group_of[j] == g]
        w = np.array([sizes[j] for j in idx], float)
        v = (C[idx] * w[:, None]).sum(0) / w.sum()
        return v / (np.linalg.norm(v) + 1e-8)

    cA, cB = group_centroid(a_g), group_centroid(b_g)

    # Map the two team groups onto classifier ids by majority overlap
    # (evaluation-only step; the grouping above never sees stored labels).
    def overlap(g, team):
        return sum(1 for t, l in zip(tids, lab)
                   if group_of[l] == g and stored[t] == team)

    direct = overlap(a_g, 0) + overlap(b_g, 1)
    flipped = overlap(a_g, 1) + overlap(b_g, 0)
    g_team = ({a_g: 0, b_g: 1} if direct >= flipped else {a_g: 1, b_g: 0})

    in_team = [(t, l) for t, l in zip(tids, lab)
               if group_of[l] in team_groups and stored[t] in (0, 1)]
    agree = sum(1 for t, l in in_team if g_team[group_of[l]] == stored[t])
    n_other = sum(1 for l in lab if group_of[l] not in team_groups)

    simA = X @ cA
    simB = X @ cB
    result = {}
    for i, t in enumerate(tids):
        g = group_of[lab[i]]
        best_team = g_team[a_g] if simA[i] >= simB[i] else g_team[b_g]
        result[t] = {
            "cluster": int(lab[i]),
            "cluster_team": g_team.get(g),        # None = other/refs group
            "nearest_team": int(best_team),
            "margin": float(abs(simA[i] - simB[i])),
            "stored": stored[t],
        }
    stats = {"n": len(in_team), "agree": agree,
             "agreement": agree / len(in_team) if in_team else float("nan"),
             "n_nonteam_cluster": n_other,
             "n_tracks": len(tids)}
    if verbose:
        print(f"[{slug} p{period}] tracks={len(tids):4d} team-grouped="
              f"{len(in_team):4d} agreement={stats['agreement']*100:5.1f}% "
              f"other-group holds {n_other} tracks")
    return {"per_track": result, "stats": stats}


def evaluate(matches=None) -> None:
    base = Config.OUTPUT_GAME_STATE_DIR if hasattr(Config, "OUTPUT_GAME_STATE_DIR") \
        else Config.PROJECT_ROOT / "output" / "game_state"
    slugs = matches or sorted(p.name for p in base.iterdir() if p.is_dir())
    tot_n = tot_agree = 0
    for slug in slugs:
        for period in available_periods(slug):
            r = cluster_half(slug, period)
            if not r:
                continue
            tot_n += r["stats"]["n"]
            tot_agree += r["stats"]["agree"]
    if tot_n:
        print("=" * 60)
        print(f"POOLED agreement vs human-labeled classifier: "
              f"{tot_agree}/{tot_n} = {tot_agree/tot_n*100:.1f}%")
        print("(high = OSNet clustering reproduces supervised teams "
              "unsupervised; per-half dips = candidate classifier errors)")


def apply_repair(slug: str, period: int, margin: float = APPLY_MARGIN) -> int:
    """Flip stored team_id for tracks whose cluster team CONFIDENTLY
    disagrees with the stored label. Backs up players.parquet first.
    Returns number of flipped tracks."""
    from .roles import infer_attack_direction, identify_goalkeepers
    r = cluster_half(slug, period, verbose=False)
    if not r:
        print(f"[{slug} p{period}] nothing to apply")
        return 0
    gs = GameState.load(slug, period=period)
    gk = set(identify_goalkeepers(gs, infer_attack_direction(gs)))
    flips = {}
    for t, info in r["per_track"].items():
        if t in gk or info["stored"] not in (0, 1):
            continue
        ct = info["cluster_team"]
        if ct is None:                       # ref/other cluster — don't touch
            continue
        if ct != info["stored"] and info["nearest_team"] == ct \
                and info["margin"] >= margin:
            flips[t] = ct
    if not flips:
        print(f"[{slug} p{period}] no confident flips "
              f"(agreement {r['stats']['agreement']*100:.1f}%)")
        return 0
    pdir = game_state_dir(slug, period)
    src = pdir / "players.parquet"
    bak = pdir / "players.pre_team_cluster.parquet"
    if not bak.exists():
        shutil.copy2(src, bak)
    pl = pd.read_parquet(src)
    mask = pl.track_id.isin(list(flips))
    pl.loc[mask, "team_id"] = pl.loc[mask, "track_id"].map(flips)
    pl.to_parquet(src)
    print(f"[{slug} p{period}] flipped {len(flips)} tracks "
          f"({int(mask.sum())} rows), backup at {bak.name}")
    return len(flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=None)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--margin", type=float, default=APPLY_MARGIN)
    args = ap.parse_args()
    if args.eval:
        evaluate([args.match] if args.match else None)
    elif args.apply and args.match:
        periods = [args.half] if args.half else available_periods(args.match)
        for p in periods:
            apply_repair(args.match, p, margin=args.margin)
    elif args.match:
        periods = [args.half] if args.half else available_periods(args.match)
        for p in periods:
            cluster_half(args.match, p)
    else:
        ap.error("use --eval, or --match X [--half N] [--apply]")


if __name__ == "__main__":
    main()
