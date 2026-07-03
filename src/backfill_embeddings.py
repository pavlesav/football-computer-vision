"""
Backfill per-track appearance embeddings for artifacts that predate their
persistence in :mod:`src.pipeline`.

Cross-half player ReID needs mean ResNet embeddings per track
(``embeddings.npz`` in each artifact dir). Newer perception runs persist them;
older artifacts (sut-mla p1, bud-sut, dec-mla) don't — but they *do* store
every player bbox, so the embeddings can be recomputed from the video without
YOLO, PnLCalib, or tracking: decode the sampled frames, mask + embed the
stored boxes with the same ``TeamClassifier`` machinery the pipeline uses.
~10 min for a full half vs ~4 h for a perception re-run.

Usage::

    python -m src.backfill_embeddings --match sut-mla --half 1
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np

from .config import Config
from .game_state import GameState
from .team_classifier import TeamClassifier

SAMPLE_EVERY = 15      # same cadence as pipeline.TEAM_SAMPLE_EVERY


def backfill(slug: str, period: int = None, force: bool = False,
             sample_every: int = SAMPLE_EVERY) -> dict:
    gs = GameState.load(slug, period=period)
    out_path = gs.dir / "embeddings.npz"
    if out_path.exists() and not force:
        print(f"[{slug} p{gs.period}] embeddings.npz already exists - skipping "
              f"(--force to redo)")
        return {}

    clf = TeamClassifier.load(
        Config.OUTPUT_CLASSIFIERS_DIR / f"{slug}_classifier.pkl")

    start = int(gs.meta["start_frame"])
    # Sampled frames on the pipeline's cadence, restricted to frames that
    # actually have players.
    pl = gs.players
    pl = pl[((pl["frame"] - start) % sample_every == 0)]
    by_frame = {int(f): g for f, g in pl.groupby("frame")}
    frames = sorted(by_frame)
    print(f"[{slug} p{gs.period}] embedding {len(pl)} boxes over "
          f"{len(frames)} sampled frames...")

    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[slug]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0] if frames else start)
    cur = frames[0] if frames else start

    track_embeddings: dict[int, list] = defaultdict(list)
    done = 0
    for f in frames:
        while cur < f:
            cap.grab()
            cur += 1
        ok, img = cap.read()
        cur += 1
        if not ok:
            break
        rows = by_frame[f]
        players = [{"bbox": [int(r.x1), int(r.y1), int(r.x2), int(r.y2)]}
                   for r in rows.itertuples(index=False)]
        tids = [int(r.track_id) for r in rows.itertuples(index=False)]
        embs = clf.collect_embeddings(img, players)
        for tid, e in zip(tids, embs):
            if np.linalg.norm(e) > 0:
                track_embeddings[tid].append(e)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(frames)} frames")
    cap.release()

    emb_ids = sorted(track_embeddings)
    emb_mat = np.stack([np.mean(track_embeddings[t], axis=0)
                        for t in emb_ids]) if emb_ids else np.zeros((0, 1))
    np.savez_compressed(out_path, track_ids=np.array(emb_ids),
                        embeddings=emb_mat)
    print(f"[{slug} p{gs.period}] wrote {len(emb_ids)} track embeddings -> "
          f"{out_path}")
    return track_embeddings


def main():
    ap = argparse.ArgumentParser(description="Backfill per-track embeddings from stored bboxes")
    ap.add_argument("--match", default="sut-mla")
    ap.add_argument("--half", type=int, default=None, choices=[1, 2])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    backfill(args.match, period=args.half, force=args.force)


if __name__ == "__main__":
    main()
