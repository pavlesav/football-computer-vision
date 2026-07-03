"""
One-command full-match processing: perception for both halves → stabilize →
goal oracle → merged StatsBomb events → one-page report.

Designed for overnight batch use, so it is **resumable**: every step records
completion in ``output/match_runs/{slug}_status.json`` and finished artifacts
are detected and skipped — a crashed or killed run continues where it left
off (perception itself also checkpoints partial artifacts every 15k frames).

Launch DETACHED from any interactive session (a harness-tied background job
dies with the session; an orphaned python may survive but nothing should rely
on that)::

    # PowerShell — one match, fully detached, logs to output/match_runs/
    Start-Process -WindowStyle Hidden -FilePath python `
        -ArgumentList "-m","src.run_match","--match","sut-mla","--home_team","1" `
        -RedirectStandardOutput output/match_runs/sut-mla.out `
        -RedirectStandardError  output/match_runs/sut-mla.err

``--home_team`` is the classifier team id of the home side (left team on the
broadcast graphics) — verify per match via kit colors before batching.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .game_state import game_state_dir

PY = sys.executable


def _load_period_info(slug: str) -> dict:
    periods = json.loads((Config.PROJECT_ROOT / "data" /
                          "period_detection_results.json").read_text())
    return next(r for r in periods if r["slug"] == slug)


def _status_path(slug: str) -> Path:
    d = Config.OUTPUT_DIR / "match_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{slug}_status.json"


def _load_status(slug: str) -> dict:
    p = _status_path(slug)
    return json.loads(p.read_text()) if p.exists() else {}


def _mark(slug: str, step: str) -> None:
    st = _load_status(slug)
    st[step] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _status_path(slug).write_text(json.dumps(st, indent=2))


def _run(slug: str, step: str, args: list) -> None:
    print(f"[{slug}] {step}: {' '.join(args)}", flush=True)
    r = subprocess.run([PY, "-m", *args], cwd=str(Config.PROJECT_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"{step} failed with exit code {r.returncode}")
    _mark(slug, step)


def _artifact_complete(slug: str, period: int) -> bool:
    meta_p = game_state_dir(slug, period) / "meta.json"
    if not meta_p.exists():
        return False
    meta = json.loads(meta_p.read_text())
    if meta.get("partial"):
        return False
    span = meta.get("end_frame", 0) - meta.get("start_frame", 0)
    return meta.get("n_frames", 0) >= 0.95 * span


def process_match(slug: str, home_team: int, pnl_stride: int = 3,
                  skip_perception: bool = False) -> None:
    info = _load_period_info(slug)
    fps = float(info["fps"])
    fh, sh = info["first_half_start_frame"], info["second_half_start_frame"]
    total = info["total_frames"]
    spans = {1: int((sh - fh) / fps), 2: int((total - sh) / fps) + 1}
    status = _load_status(slug)

    for period in (1, 2):
        step = f"perception_p{period}"
        if skip_perception or _artifact_complete(slug, period):
            print(f"[{slug}] {step}: artifact complete - skip", flush=True)
        else:
            _run(slug, step, ["src.pipeline", "--match", slug,
                              "--half", str(period), "--offset_min", "0",
                              "--duration_sec", str(spans[period]),
                              "--pnl_stride", str(pnl_stride)])
        step = f"stabilize_p{period}"
        if status.get(step):
            print(f"[{slug}] {step}: done - skip", flush=True)
        else:
            _run(slug, step, ["src.stabilize", "--match", slug,
                              "--half", str(period), "--apply"])
        step = f"jersey_p{period}"
        if status.get(step) or (game_state_dir(slug, period)
                                / "jersey_numbers.json").exists():
            print(f"[{slug}] {step}: done - skip", flush=True)
        else:
            # CPU-only (easyocr); ~1h per half at the default crop budget.
            _run(slug, step, ["src.jersey_ocr", "--match", slug,
                              "--half", str(period)])

    if status.get("score_ocr"):
        print(f"[{slug}] score_ocr: done - skip", flush=True)
    else:
        _run(slug, "score_ocr", ["src.score_ocr", "--match", slug,
                                 "--home_team", str(home_team)])

    # Events + report are cheap — always rebuilt so they reflect the latest
    # event logic.
    _run(slug, "events", ["src.events", "--match", slug])
    _run(slug, "report", ["src.report", "--match", slug])
    print(f"[{slug}] MATCH COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Process one match end to end (resumable)")
    ap.add_argument("--match", required=True)
    ap.add_argument("--home_team", type=int, required=True, choices=[0, 1],
                    help="classifier team id of the HOME side (verify via "
                         "kit colors, e.g. sut-mla: 1)")
    ap.add_argument("--pnl_stride", type=int, default=3)
    ap.add_argument("--skip_perception", action="store_true",
                    help="only run the analysis chain on existing artifacts")
    args = ap.parse_args()
    process_match(args.match, args.home_team, pnl_stride=args.pnl_stride,
                  skip_perception=args.skip_perception)


if __name__ == "__main__":
    main()
