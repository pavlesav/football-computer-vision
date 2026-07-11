"""Resumable VLM jersey rollout: run Qwen2-VL extraction on every half that
doesn't already have a qwen2-vl jersey_numbers.json, backing up the easyocr
file first. Safe to relaunch after an interruption — completed halves skip.

Run (detached):
    python scripts/vlm_rollout.py --matches dec-mla jed-ars jez-ars jez-jed mla-bud-2 sut-mla
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game_state import available_periods                     # noqa: E402
from src.jersey_ocr import jersey_path                           # noqa: E402
from src import jersey_vlm                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--max_crops", type=int, default=jersey_vlm.VLM_MAX_CROPS)
    ap.add_argument("--min_box_h", type=int, default=jersey_vlm.VLM_MIN_BOX_H)
    args = ap.parse_args()

    for slug in args.matches:
        for period in available_periods(slug):
            p = jersey_path(slug, period)
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("params", {}).get("reader") == "qwen2-vl":
                    print(f"[{slug} p{period}] already qwen2-vl — skip",
                          flush=True)
                    continue
                bak = p.with_suffix(".easyocr.json")
                if not bak.exists():
                    shutil.copy2(p, bak)
                    print(f"[{slug} p{period}] backed up easyocr file",
                          flush=True)
            jersey_vlm.extract(slug, period, max_crops=args.max_crops,
                               force=True, min_box_h=args.min_box_h)
    print("ROLLOUT DONE", flush=True)


if __name__ == "__main__":
    main()
