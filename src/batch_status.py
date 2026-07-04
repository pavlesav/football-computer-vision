"""
Readable batch status — the antidote to a 3MB per-frame log.

Prints, for the Tier-1 queue: each match's completed steps (from the resumable
status JSONs), and for the match currently in perception, the latest progress
percent + ETA parsed from the *tail* of the log (never reads the whole file).

    python -m src.batch_status
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Config
from .run_batch import TIER1_QUEUE

STEPS = ["perception_p1", "stabilize_p1", "jersey_p1",
         "perception_p2", "stabilize_p2", "jersey_p2",
         "score_ocr", "events", "report"]
LOG = Config.OUTPUT_DIR / "match_runs" / "batch_tier1.err"
# last "<slug> perception:  64%|... | 47898/75275 [3:12:04<5:22:52, 1.41it/s]"
_PROG = re.compile(
    r"([\w-]+) perception:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*"
    r"\[([\d:]+)<([\d:?]+),\s*([\d.]+)it/s")


def _tail(path: Path, nbytes: int = 8192) -> str:
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", "replace")


def _last_progress() -> dict | None:
    if not LOG.exists():
        return None
    m = None
    for m in _PROG.finditer(_tail(LOG)):
        pass                     # keep the last match in the tail window
    if not m:
        return None
    slug, pct, done, total, elapsed, eta, rate = m.groups()
    return {"slug": slug, "pct": int(pct), "done": int(done),
            "total": int(total), "elapsed": elapsed, "eta": eta,
            "rate": float(rate)}


def main() -> None:
    prog = _last_progress()
    running = prog["slug"] if prog else None
    print(f"{'match':<12} {'done':>3}/9  steps")
    for slug, home in TIER1_QUEUE:
        st = {}
        p = Config.OUTPUT_DIR / "match_runs" / f"{slug}_status.json"
        if p.exists():
            st = json.loads(p.read_text())
        done = [s for s in STEPS if s in st]
        if len(done) == len(STEPS):
            mark = "DONE"
        elif slug == running:
            mark = "RUNNING"
        elif done:
            mark = "partial"
        else:
            mark = "queued"
        next_step = next((s for s in STEPS if s not in st), "-")
        print(f"{slug:<12} {len(done):>3}/9  {mark:<8} next: {next_step}")

    if prog:
        print(f"\nnow: {prog['slug']} perception {prog['pct']}% "
              f"({prog['done']}/{prog['total']} frames) - "
              f"elapsed {prog['elapsed']}, ETA {prog['eta']} "
              f"@ {prog['rate']:.1f} it/s")
    else:
        print("\nno active perception in the log tail "
              "(between steps, or batch finished)")


if __name__ == "__main__":
    main()
