"""
Appearance re-identification (OSNet-AIN) over stored player bboxes.

Why this exists: player-level stats are capped by track fragmentation (BoT-SORT
restarts a track at every camera cut). The two automatic bridges we had both
stall — kinematic consolidation / :mod:`src.identity_propagation` cannot cross a
cut, and the persisted team-classifier **ResNet18** embeddings were *measured* to
carry no within-team identity signal (P(same-player pair closer) = 0.43 = chance,
see CLAUDE.md). The SoccerNet-2025 GSR winner fixes exactly this with a
*purpose-trained* appearance-ReID model (OSNet) + a split-and-merge tracklet
post-process. This module is step one: a purpose-trained ReID embedding computed
offline over the persisted game state, and a **decisive discriminative test** of
whether it beats the ResNet18 result on OUR footage before any merge logic is
built on it.

Model: ``osnet_ain_x1_0`` trained multi-source for domain **generalization**
(MS+D+C->M, 73.3 mAP) — the torchreid variant recommended for unseen domains,
which broadcast 1.CFL soccer is relative to person-ReID datasets. Weights live at
``models/reid/osnet_ain_x1_0_clean.pth`` (state-dict-only re-save; the raw model-
zoo checkpoint carries numpy scalars that trip torch>=2.6 weights_only loading).

Discriminative test (``--eval``): ground truth is the human review-UI labels in
``data/identities/{slug}_p{N}.json`` (raw-track -> shirt number + team; metas the
human flagged as mixed are excluded). Within each half we form **same-player**
track pairs (same team+number, different track) and **cross** pairs (same team,
different number) and report P(same-pair cosine > cross-pair cosine) — the same
AUC-style number that scored 0.43 for ResNet18. Gate: >= ~0.70 justifies building
the Phase-3 split-and-merge; near-chance kills it.

Run::

    python -m src.reid --eval                 # pooled over all identity files
    python -m src.reid --eval --match sut-mla  # one match, verbose per-half
    python -m src.reid --match sut-mla --half 1  # cache reid.npz for all tracks
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .game_state import GameState, game_state_dir

# ── Model ─────────────────────────────────────────────────────────────────
REID_MODEL_NAME = "osnet_ain_x1_0"
REID_WEIGHTS = Config.PROJECT_ROOT / "models" / "reid" / "osnet_ain_x1_0_clean.pth"

# ── Crop / sampling tunables ──────────────────────────────────────────────
MIN_TRACK_ROWS = 25       # ignore flicker tracks
MIN_BOX_H = 60            # px; ReID needs a body, not a legible number (< jersey's 90)
MAX_CROPS_PER_TRACK = 12
FRAME_DEDUP_GAP = 20      # frames; spread anchors across the track's lifetime
BATCH = 256

_EXTRACTOR = None


def _extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        import warnings
        warnings.filterwarnings("ignore")
        from torchreid.reid.utils import FeatureExtractor
        if not REID_WEIGHTS.exists():
            raise FileNotFoundError(
                f"ReID weights not found at {REID_WEIGHTS}. Download osnet_ain_x1_0 "
                f"and re-save a clean state-dict (see module docstring).")
        _EXTRACTOR = FeatureExtractor(
            model_name=REID_MODEL_NAME, model_path=str(REID_WEIGHTS),
            device="cuda", verbose=False)
    return _EXTRACTOR


# ── Crop planning ─────────────────────────────────────────────────────────

def _anchor_frames(sub, max_n: int = MAX_CROPS_PER_TRACK,
                   min_h: int = MIN_BOX_H, gap: int = FRAME_DEDUP_GAP) -> list:
    """Up to ``max_n`` rows of one track: tallest boxes first, skipping any
    within ``gap`` frames of an already-picked one so the sample spreads over
    the track's lifetime (mirrors :func:`jersey_ocr._pick_frames`)."""
    tall = sub[sub.h >= min_h].sort_values("h", ascending=False)
    picks, picked = [], []
    for r in tall.itertuples(index=False):
        f = int(r.frame)
        if any(abs(f - pf) < gap for pf in picked):
            continue
        picks.append(r)
        picked.append(f)
        if len(picks) >= max_n:
            break
    return picks


def _crop_rgb(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Full-body bbox crop, clamped, BGR->RGB (OSNet transforms expect RGB)."""
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def embed_tracks(slug: str, period: int, track_ids,
                 gs: Optional[GameState] = None,
                 max_per_track: int = MAX_CROPS_PER_TRACK,
                 min_box_h: int = MIN_BOX_H, verbose: bool = False) -> dict:
    """{track_id: L2-normalized mean OSNet embedding (float32[512])} for the
    given tracks. Reads only the anchor frames (per-anchor seek, not decode of
    the whole half), so a few dozen tracks embed in seconds."""
    gs = gs or GameState.load(slug, period=period)
    want = set(int(t) for t in track_ids)
    pl = gs.players[gs.players.track_id.isin(want)].copy()
    if pl.empty:
        return {}
    pl["h"] = pl.y2 - pl.y1

    jobs = []   # (track_id, frame, x1,y1,x2,y2)
    for tid, g in pl.groupby("track_id"):
        for r in _anchor_frames(g, max_per_track, min_box_h):
            jobs.append((int(tid), int(r.frame),
                         int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
    if not jobs:
        return {}
    jobs.sort(key=lambda j: j[1])   # ascending frame -> forward-only seeks

    video = str(Config.MATCH_VIDEOS[slug])
    cap = cv2.VideoCapture(video)
    crops, crop_tids = [], []
    t0 = time.time()
    for i, (tid, f, x1, y1, x2, y2) in enumerate(jobs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        c = _crop_rgb(img, x1, y1, x2, y2)
        if c is None:
            continue
        crops.append(c)
        crop_tids.append(tid)
        if verbose and (i + 1) % 500 == 0:
            print(f"    read {i+1}/{len(jobs)} crops "
                  f"({(time.time()-t0):.0f}s)")
    cap.release()
    if not crops:
        return {}

    ex = _extractor()
    feats = []
    for s in range(0, len(crops), BATCH):
        f = ex(crops[s:s + BATCH]).cpu().numpy()
        feats.append(f)
    feats = np.concatenate(feats, 0).astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    out: dict = {}
    tids = np.array(crop_tids)
    for tid in np.unique(tids):
        m = feats[tids == tid].mean(0)
        m /= (np.linalg.norm(m) + 1e-8)
        out[int(tid)] = m.astype(np.float32)
    return out


# ── Persistence (for Phase 3 reuse) ───────────────────────────────────────

def reid_path(slug: str, period: int) -> Path:
    return game_state_dir(slug, period) / "reid.npz"


def save_embeddings(slug: str, period: int, embs: dict) -> Path:
    p = reid_path(slug, period)
    p.parent.mkdir(parents=True, exist_ok=True)
    tids = np.array(sorted(embs), dtype=np.int64)
    mat = np.stack([embs[int(t)] for t in tids]) if len(tids) else \
        np.zeros((0, 512), np.float32)
    np.savez_compressed(p, track_ids=tids, embeddings=mat.astype(np.float32))
    return p


def load_embeddings(slug: str, period: int) -> Optional[dict]:
    p = reid_path(slug, period)
    if not p.exists():
        return None
    d = np.load(p)
    return {int(t): e for t, e in zip(d["track_ids"], d["embeddings"])}


def relevant_tracks(slug: str, period: int, gs: Optional[GameState] = None) -> list:
    """Tracks worth embedding for identity propagation: the ones that BEAR
    events (carrier/recipient) plus the human anchors. A half has ~4000
    substantial fragments but only ~400 bear events — the rest are 1-second
    noise that no label needs to reach — so this is ~10x cheaper than embedding
    everything and loses nothing that affects attribution."""
    gs = gs or GameState.load(slug, period=period)
    from .events import detect_events
    from . import identity_propagation as ip
    events, _ = detect_events(gs)
    part = set()
    for e in events:
        for t in (e.player, e.details.get("recipient")):
            if t is not None and int(t) >= 0:
                part.add(int(t))
    anchors = set(ip._anchor_map_from_identity(gs, slug, period))
    j_anchors, _ = ip._jersey_meta_anchors(gs, slug, period)
    return sorted(part | anchors | set(j_anchors))


def build_all(slug: str, period: int, force: bool = False,
              relevant_only: bool = True) -> dict:
    """Embed the identity-relevant tracks of a half and cache reid.npz."""
    if not force and reid_path(slug, period).exists():
        print(f"[{slug} p{period}] reid.npz exists — skipping (--force to redo)")
        return load_embeddings(slug, period)
    gs = GameState.load(slug, period=period)
    if relevant_only:
        tracks = relevant_tracks(slug, period, gs)
        label = "event-bearing+anchor"
    else:
        counts = gs.players.groupby("track_id").size()
        tracks = [int(t) for t in counts[counts >= MIN_TRACK_ROWS].index]
        label = "substantial"
    print(f"[{slug} p{period}] embedding {len(tracks)} {label} tracks…")
    # 8 crops/track is plenty for a stable mean ReID vector and roughly halves
    # the video-seek cost vs the 12 used for the discriminative test.
    embs = embed_tracks(slug, period, tracks, gs=gs, verbose=True,
                        max_per_track=8)
    p = save_embeddings(slug, period, embs)
    print(f"[{slug} p{period}] cached {len(embs)} track embeddings -> {p}")
    return embs


# ── Discriminative test (the Phase-1 gate) ────────────────────────────────

def _labeled_tracks(identity_file: Path) -> dict:
    """{track_id: (team, number)} for cleanly-labeled tracks in one half
    (mixed-flagged metas excluded)."""
    d = json.loads(identity_file.read_text(encoding="utf-8"))
    mixed = set(int(x) for x in d.get("mixed_metas", []))
    out = {}
    for k, v in d["players"].items():
        num = v.get("number")
        if num is None or int(k) in mixed:
            continue
        out[int(k)] = (int(v["team"]), int(num))
    return out


def _auc_same_gt_cross(same_cos, cross_cos) -> float:
    """P(random same-pair cosine > random cross-pair cosine) = Mann-Whitney AUC.
    Ties count 0.5. Rank-based, O((n+m) log(n+m))."""
    same = np.asarray(same_cos); cross = np.asarray(cross_cos)
    if len(same) == 0 or len(cross) == 0:
        return float("nan")
    allv = np.concatenate([same, cross])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_same = ranks[:len(same)].sum()
    auc = (r_same - len(same) * (len(same) + 1) / 2.0) / (len(same) * len(cross))
    return float(auc)


def discriminative_test(matches: Optional[list] = None,
                        verbose: bool = True) -> dict:
    """Within-half P(same>cross) for OSNet ReID embeddings, pooled across all
    identity files. Prints per-half and pooled numbers; returns the summary."""
    base = Config.PROJECT_ROOT / "data" / "identities"
    files = sorted(base.glob("*.json"))
    if matches:
        files = [f for f in files if any(f.name.startswith(m + "_") for m in matches)]

    all_same, all_cross = [], []           # pooled within-half cosines
    per_half_auc = []
    tot_same = tot_cross = 0
    for f in files:
        stem = f.stem                       # e.g. sut-mla_p1
        slug, per = stem.rsplit("_p", 1)
        period = int(per)
        labels = _labeled_tracks(f)
        if len(labels) < 4:
            continue
        embs = embed_tracks(slug, period, list(labels))
        labels = {t: lab for t, lab in labels.items() if t in embs}   # embedded only

        # group by (team, number) -> same-player track sets
        by_player = defaultdict(list)
        for t, (team, num) in labels.items():
            by_player[(team, num)].append(t)

        def cos(a, b):
            return float(embs[a] @ embs[b])

        same, cross = [], []
        players = list(by_player)
        # same-player pairs
        for ts in by_player.values():
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    same.append(cos(ts[i], ts[j]))
        # cross pairs: same team, different number
        by_team = defaultdict(list)
        for (team, num), ts in by_player.items():
            by_team[team].append(ts)
        for team, groups in by_team.items():
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    for a in groups[gi]:
                        for b in groups[gj]:
                            cross.append(cos(a, b))
        if not same or not cross:
            continue
        auc = _auc_same_gt_cross(same, cross)
        per_half_auc.append((auc, len(same), len(cross)))
        all_same += same; all_cross += cross
        tot_same += len(same); tot_cross += len(cross)
        if verbose:
            print(f"{stem:<20} tracks={len(labels):3d} same={len(same):4d} "
                  f"cross={len(cross):5d} | mean cos same={np.mean(same):.3f} "
                  f"cross={np.mean(cross):.3f} | P(same>cross)={auc:.3f}")

    if not all_same:
        print("no labeled pairs found")
        return {}
    pooled_auc = _auc_same_gt_cross(all_same, all_cross)
    # weighted mean of per-half AUCs (each half's cosine scale is its own)
    w = np.array([s * c for _, s, c in per_half_auc], float)
    mean_half_auc = float(np.average([a for a, _, _ in per_half_auc], weights=w))
    ms, mc = np.mean(all_same), np.mean(all_cross)
    sd = np.sqrt((np.var(all_same) + np.var(all_cross)) / 2)
    cohen_d = (ms - mc) / (sd + 1e-8)
    print("\n" + "=" * 66)
    print(f"OSNet-AIN ReID discriminative test — {len(per_half_auc)} halves")
    print(f"  same-player pairs : {tot_same}   cross pairs : {tot_cross}")
    print(f"  mean cosine        same={ms:.3f}  cross={mc:.3f}  "
          f"(gap {ms-mc:+.3f}, Cohen d {cohen_d:+.2f})")
    print(f"  P(same>cross) pooled        = {pooled_auc:.3f}")
    print(f"  P(same>cross) mean-per-half = {mean_half_auc:.3f}")
    print(f"  [ResNet18 baseline was 0.43 = chance; GATE >= 0.70]")
    verdict = "PASS -> build Phase 3 split-and-merge" if mean_half_auc >= 0.70 \
        else ("MARGINAL" if mean_half_auc >= 0.60 else "FAIL -> ReID dead end here")
    print(f"  VERDICT: {verdict}")
    return {"pooled_auc": pooled_auc, "mean_half_auc": mean_half_auc,
            "mean_same": float(ms), "mean_cross": float(mc),
            "cohen_d": float(cohen_d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=None)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--eval", action="store_true",
                    help="run the P(same>cross) discriminative test")
    ap.add_argument("--all_tracks", action="store_true",
                    help="embed every substantial track (default: event-bearing+anchor)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.eval:
        discriminative_test(matches=[args.match] if args.match else None)
    elif args.match and args.half:
        build_all(args.match, args.half, force=args.force,
                  relevant_only=not args.all_tracks)
    else:
        ap.error("use --eval, or --match X --half N to cache reid.npz")


if __name__ == "__main__":
    main()
