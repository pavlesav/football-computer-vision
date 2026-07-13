"""Does image enhancement rescue PnLCalib on night-game frames?

The two stuck matches (mla-bud-2 24% trusted, jed-ars 46%) are floodlit:
paint contrast collapses and PnLCalib misses lines it finds easily in
daylight. Before committing to the multi-day fine-tune, measure the cheap
alternative: preprocess the frame (CLAHE on luminance / gamma lift /
combination) and re-run the SAME pretrained model + gates.

Protocol: sample N wide-shot gameplay frames the artifact recorded WITHOUT
a fresh PnLCalib success, plus N/3 control frames where PnLCalib succeeded
(enhancement must not break those). For each frame x variant, run
predict_one_frame (full gates). Report gate-pass rate per variant.

Run: python scripts/night_enhance_test.py --match mla-bud-2 --half 1 --n 30
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config                                  # noqa: E402
from src.game_state import GameState                           # noqa: E402
from src.run_pnlcalib_video import load_models, predict_one_frame  # noqa: E402


def clahe_l(img, clip=2.5, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def gamma(img, g=0.6):
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


VARIANTS = {
    "raw": lambda f: f,
    "clahe": lambda f: clahe_l(f),
    "gamma06": lambda f: gamma(f, 0.6),
    "clahe+gamma": lambda f: gamma(clahe_l(f), 0.7),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    gs = GameState.load(args.match, period=args.half)
    fr = gs.frames
    wide = fr[(fr.is_gameplay == True) & (fr.is_wide_shot == True)]  # noqa: E712
    failed = wide[fr.homog_source != "pnlcalib"]
    ok = wide[fr.homog_source == "pnlcalib"]
    rng = np.random.default_rng(0)
    f_fail = sorted(rng.choice(failed.frame.values,
                               min(args.n, len(failed)), replace=False))
    f_ok = sorted(rng.choice(ok.frame.values,
                             min(args.n // 3, len(ok)), replace=False))
    print(f"[{args.match} p{args.half}] wide gameplay frames: {len(wide)} "
          f"({len(ok)} pnlcalib-fresh, {len(failed)} not) — testing "
          f"{len(f_fail)} failed + {len(f_ok)} control")

    boxes_of = {}
    pl = gs.players
    for f in list(f_fail) + list(f_ok):
        sub = pl[pl.frame == f]
        boxes_of[int(f)] = [(r.x1, r.y1, r.x2, r.y2)
                            for r in sub.itertuples()]

    models = load_models("cuda:0")
    cap = cv2.VideoCapture(str(Config.MATCH_VIDEOS[args.match]))
    results = {v: {"fail_pass": 0, "ok_pass": 0} for v in VARIANTS}
    for group, frames in (("fail", f_fail), ("ok", f_ok)):
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            r, img = cap.read()
            if not r:
                continue
            h, w = img.shape[:2]
            for vname, fn in VARIANTS.items():
                P, status = predict_one_frame(fn(img), models, w, h,
                                              player_boxes=boxes_of[int(f)])
                if P is not None and status == "pnlcalib":
                    results[vname][f"{group}_pass"] += 1
    cap.release()

    print(f"\n{'variant':<14} {'failed frames rescued':>22} {'controls kept':>15}")
    for v, r in results.items():
        print(f"{v:<14} {r['fail_pass']:>10}/{len(f_fail):<11} "
              f"{r['ok_pass']:>7}/{len(f_ok)}")


if __name__ == "__main__":
    main()
