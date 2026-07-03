"""
Sequential batch driver: full-match processing for the Tier-1 queue, one
match at a time (single GPU). Each match is handled by
:mod:`src.run_match`, which is resumable — if this driver dies, relaunch it
and completed steps are skipped.

``--home_team`` mappings below were verified visually (2026-07-03) from the
scoreboard color chips + labeled classifier kit crops, cross-validated by
kit consistency across fixtures (Mladost yellow, Sutjeska blue, Jedinstvo
white, Jezero blue, Budućnost blue-white stripes).

LAUNCH DETACHED (survives the terminal/session)::

    powershell -Command "Start-Process -WindowStyle Hidden `
      -FilePath 'C:/Users/PC/AppData/Local/Python/bin/python.exe' `
      -ArgumentList '-m','src.run_batch' `
      -WorkingDirectory 'C:/Users/PC/Desktop/GitHub/football-computer-vision' `
      -RedirectStandardOutput 'output/match_runs/batch_tier1.out' `
      -RedirectStandardError 'output/match_runs/batch_tier1.err'"

Progress: ``output/match_runs/batch_tier1.out`` + per-match status JSONs.
"""
from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone

from .run_match import process_match

# (slug, home_team classifier id) — verified, do not guess new entries.
TIER1_QUEUE = [
    ("jed-ars", 1),
    ("jez-ars", 0),
    ("jez-jed", 0),
    ("sut-pet", 0),
    ("mla-bud-2", 1),
    ("dec-mla", 1),
]


def main():
    ap = argparse.ArgumentParser(description="Batch-process the Tier-1 match queue")
    ap.add_argument("--only", default=None,
                    help="comma-separated slug subset of the queue")
    args = ap.parse_args()
    queue = TIER1_QUEUE
    if args.only:
        want = set(args.only.split(","))
        queue = [q for q in queue if q[0] in want]

    for slug, home in queue:
        t = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"\n########## {t} BATCH: {slug} (home_team {home}) ##########",
              flush=True)
        try:
            process_match(slug, home)
        except Exception:
            print(f"[{slug}] FAILED - continuing with next match", flush=True)
            traceback.print_exc()
    print("\n########## BATCH QUEUE DONE ##########", flush=True)


if __name__ == "__main__":
    main()
