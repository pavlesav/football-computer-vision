"""Contact sheets for golden-event labeling (vision-model or human).

Draws every player bbox with its TRACK ID (team-colored) plus all raw ball
detections (magenta circle) on every Nth frame of a window, stacked into
strip images. The labeler steps through the strips and writes the golden
JSON (control intervals + passes + unknown_spans) by hand.

Also prints a window-quality scan (--scan) so the labeled segment sits in
a stretch with trusted homography and dense ball evidence — golden windows
should test the chain where it can actually see.

Run:
    python scripts/golden_label_sheets.py --match jez-jed --half 1 --scan
    python scripts/golden_label_sheets.py --match jez-jed --half 1 \
        --f0 18000 --f1 19200 --step 8 --out DIR
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config                      # noqa: E402
from src.game_state import GameState, trusted_frame_mask  # noqa: E402

TEAM_BGR = {0: (60, 220, 255), 1: (255, 160, 60)}


def scan(gs, win=1250):
    fr = gs.frames
    trusted = trusted_frame_mask(gs)
    ball = gs.ball_candidates().groupby("frame").size()
    f0, f1 = int(fr.frame.min()), int(fr.frame.max())
    print("window_start  trusted%  ball-det%  (1250-frame windows, step 625)")
    best = []
    for s in range(f0, f1 - win, win // 2):
        m = fr.frame.between(s, s + win)
        t = trusted[m].mean() * 100
        b = ball[ball.index.to_series().between(s, s + win)].count() / win * 100
        best.append((t * 0.6 + b * 0.4, s, t, b))
    for score, s, t, b in sorted(best, reverse=True)[:10]:
        mm = int(s / gs.fps // 60)
        ss = int(s / gs.fps % 60)
        print(f"  f{s:<8} {t:5.0f}%    {b:5.0f}%   ({mm}:{ss:02d} into half)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--f0", type=int)
    ap.add_argument("--f1", type=int)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--per_row", type=int, default=5)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gs = GameState.load(args.match, period=args.half)
    if args.scan:
        scan(gs)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pl = gs.players
    ball = gs.ball_candidates()
    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[args.match]))
    frames = list(range(args.f0, args.f1 + 1, args.step))
    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        sub = pl[pl.frame == f]
        for r in sub.itertuples():
            c = TEAM_BGR.get(int(r.team_id), (200, 200, 200))
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)),
                          c, 2)
            cv2.putText(img, str(int(r.track_id)),
                        (int(r.x1), int(r.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(img, str(int(r.track_id)),
                        (int(r.x1), int(r.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        for b in ball[ball.frame == f].itertuples():
            cx, cy = int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2)
            cv2.circle(img, (cx, cy), 18, (255, 0, 255), 3)
        t = f / gs.fps
        cv2.putText(img, f"f{f}  {int(t//60)}:{t%60:04.1f}",
                    (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
        cv2.putText(img, f"f{f}  {int(t//60)}:{t%60:04.1f}",
                    (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (80, 255, 255), 2)
        tiles.append(cv2.resize(img, (768, 432)))
    cap.release()

    per_sheet = args.per_row * args.rows
    n_sheet = 0
    for s in range(0, len(tiles), per_sheet):
        chunk = tiles[s:s + per_sheet]
        rows = []
        for r in range(0, len(chunk), args.per_row):
            row = chunk[r:r + args.per_row]
            while len(row) < args.per_row:
                row.append(np.zeros_like(row[0]))
            rows.append(cv2.hconcat(row))
        sheet = cv2.vconcat(rows)
        cv2.imwrite(str(out / f"g{n_sheet:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
        n_sheet += 1
    print(f"{len(tiles)} frames -> {n_sheet} sheets in {out}")


if __name__ == "__main__":
    main()
