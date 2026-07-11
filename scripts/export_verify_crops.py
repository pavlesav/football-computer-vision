"""Export verify-UI cards as contact-sheet JPEGs for model-vision review.

Writes one image per card (a row of timestamped crops) plus a manifest JSON
with the card metadata (claimed team/number, events, fragments). A vision
model (or a human without a browser) reviews the sheets and produces the
same JSON that verify_ui --apply consumes.

Run: python scripts/export_verify_crops.py --match sut-pet --half 2 --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config                              # noqa: E402
from src.game_state import GameState                       # noqa: E402
from src.review_ui import _best_crop_rows                  # noqa: E402
from src.verify_ui import resolve_groups                   # noqa: E402


def sheet(gs, cap, tids, prefer, time_map, k=8, crop_h=190):
    rows = _best_crop_rows(gs, tids, k=k, prefer_frames=prefer)
    tiles = []
    for r in sorted(rows, key=lambda r: int(r.frame)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, img = cap.read()
        if not ok:
            continue
        pad = 12
        h, w = img.shape[:2]
        crop = img[max(int(r.y1) - pad, 0):min(int(r.y2) + pad, h),
                   max(int(r.x1) - pad, 0):min(int(r.x2) + pad, w)]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        crop = cv2.resize(crop, (int(cw * crop_h / ch), crop_h))
        t = time_map.get(int(r.frame))
        if t is not None and np.isfinite(t):
            lab = f"{int(t // 60)}:{int(t % 60):02d}"
            cv2.putText(crop, lab, (3, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(crop, lab, (3, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 255, 255), 1)
        tiles.append(crop)
    if not tiles:
        return None
    return cv2.hconcat([cv2.copyMakeBorder(t, 0, 0, 0, 4,
                                           cv2.BORDER_CONSTANT,
                                           value=(30, 30, 30))
                        for t in tiles])


def _label_strip(text: str, width: int, h: int = 26) -> np.ndarray:
    strip = np.full((h, width, 3), (25, 28, 34), np.uint8)
    cv2.putText(strip, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (120, 220, 255), 1, cv2.LINE_AA)
    return strip


def _stack(rows: list) -> np.ndarray:
    """Vertically stack (label, image) rows onto one canvas."""
    width = max(img.shape[1] for _, img in rows)
    parts = []
    for label, img in rows:
        if img.shape[1] < width:
            img = cv2.copyMakeBorder(img, 0, 0, 0, width - img.shape[1],
                                     cv2.BORDER_CONSTANT, value=(15, 15, 15))
        parts.append(_label_strip(label, width))
        parts.append(img)
    return cv2.vconcat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", type=int, default=0,
                    help="cards per stacked sheet (0 = one file per card)")
    ap.add_argument("--top_un", type=int, default=18,
                    help="max unattributed cards")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gs = GameState.load(args.match, period=args.half)
    groups, unattributed, read_frames = resolve_groups(gs, args.match,
                                                       args.half)
    time_map = dict(zip(gs.frames.frame.astype(int),
                        gs.frames.time_sec.astype(float)))
    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[args.match]))

    manifest = {"slug": args.match, "period": args.half, "cards": []}
    rows = []          # (label, image, card-record)
    ordered = sorted(groups.items(), key=lambda kv: -kv[1]["events"])
    ordered = [(k, g) for k, g in ordered if g["events"] > 0][:40]
    for (team, num), g in ordered:
        prefer = set()
        for t in g["tids"]:
            prefer |= read_frames.get(t, set())
        img = sheet(gs, cap, g["tids"], prefer, time_map)
        if img is None:
            continue
        rec = {"key": f"i_{team}_{num}", "kind": "identity", "team": team,
               "number": num, "events": g["events"],
               "fragments": len(g["tids"])}
        rows.append((f"i_{team}_{num}  CLAIM team{team} #{num}  "
                     f"({g['events']} ev, {len(g['tids'])} frags)", img, rec))
    for tid, n, team in [u for u in unattributed if u[1] >= 2][:args.top_un]:
        img = sheet(gs, cap, [tid], read_frames.get(tid, set()), time_map)
        if img is None:
            continue
        rec = {"key": f"u_{tid}", "kind": "track", "team": team, "events": n}
        rows.append((f"u_{tid}  team{team}  ({n} ev)", img, rec))
    cap.release()

    if args.stack > 0:
        for si in range(0, len(rows), args.stack):
            chunk = rows[si:si + args.stack]
            name = f"sheet_{si // args.stack:02d}.jpg"
            cv2.imwrite(str(out / name), _stack([(l, im) for l, im, _ in chunk]),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            for _, _, rec in chunk:
                manifest["cards"].append({**rec, "file": name})
    else:
        for label, img, rec in rows:
            name = rec["key"] + ".jpg"
            cv2.imwrite(str(out / name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            manifest["cards"].append({**rec, "file": name})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest['cards'])} cards in "
          f"{len(set(c['file'] for c in manifest['cards']))} files -> {out}")


if __name__ == "__main__":
    main()
