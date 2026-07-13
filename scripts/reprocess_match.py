"""Full re-perception + automatic identity chain for one match.

Used when the perception layer itself improved (e.g. the night-enhancement
ladder in predict_one_frame) and the artifact must be rebuilt. Track ids
restart, so every track-keyed artifact of the match is deleted and the
automatic identity stack re-derives them; human identity files are removed
too (verdicts reference dead ids) — regenerate a verify page afterwards.

Launch DETACHED (survives the session):
    Start-Process python -ArgumentList "scripts/reprocess_match.py","--match","mla-bud-2" `
        -RedirectStandardOutput output/match_runs/mla-bud-2_repro.out `
        -RedirectStandardError  output/match_runs/mla-bud-2_repro.err -WindowStyle Hidden
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config                      # noqa: E402
from src.game_state import game_state_dir          # noqa: E402

PY = sys.executable


def run(args):
    print(f">>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY, "-m", *args], cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"step failed: {args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--pnl_stride", type=int, default=3)
    args = ap.parse_args()
    slug = args.match

    periods = json.loads((ROOT / "data" /
                          "period_detection_results.json").read_text())
    info = next(r for r in periods if r["slug"] == slug)
    fps = float(info["fps"])
    fh, sh = info["first_half_start_frame"], info["second_half_start_frame"]
    spans = {1: int((sh - fh) / fps),
             2: int((info["total_frames"] - sh) / fps) + 1}

    for h in (1, 2):
        run(["src.pipeline", "--match", slug, "--half", str(h),
             "--offset_min", "0", "--duration_sec", str(spans[h]),
             "--pnl_stride", str(args.pnl_stride)])
        run(["src.stabilize", "--match", slug, "--half", str(h), "--apply"])

    # stale track-keyed artifacts: ids are dead after re-perception
    for h in (1, 2):
        d = game_state_dir(slug, h)
        for name in ("jersey_numbers.json", "reid.npz",
                     "players.pre_split.parquet"):
            p = d / name
            if p.exists():
                p.unlink()
                print(f"deleted stale {p}", flush=True)
    for h in (1, 2):
        idp = ROOT / "data" / "identities" / f"{slug}_p{h}.json"
        if idp.exists():
            idp.unlink()
            print(f"deleted stale {idp} (verdicts referenced dead ids)",
                  flush=True)

    for h in (1, 2):
        run(["src.jersey_vlm", "--match", slug, "--half", str(h), "--force"])
        run(["src.track_split", "--match", slug, "--half", str(h), "--apply"])
        run(["src.jersey_ocr", "--match", slug, "--half", str(h), "--revote"])
        run(["src.reid", "--match", slug, "--half", str(h), "--force"])
    run(["src.events", "--match", slug])
    run(["src.sofa_eval", "--match", slug])
    print("REPROCESS DONE", flush=True)


if __name__ == "__main__":
    main()
