"""
Score detected events against a hand-labeled golden set.

Golden files live in ``data/golden_events/{slug}_*.json``: ball-control
intervals + pass list (kick/receive frames ±10, participant track ids),
labeled by frame-stepping contact sheets. This turns event-logic tuning into a
measurable precision/recall problem instead of aggregate plausibility checks.

Matching rules (deliberately forgiving on timing, strict on participants):
  * a system pass matches a golden pass if the kick frames are within
    ``TOL_FRAMES`` and the passer track matches; for completed golden passes
    the receiver must match too (ids are BoT-SORT tracks in both).
  * golden passes with ``from`` ids the system cannot see (GK classified
    'Other', throw-in takers off-frame) still count as FN — they measure the
    honest gap, not a scoring bug.

Usage::

    python -m src.golden_eval --match sut-mla
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config
from .game_state import GameState
from .events import detect_events

TOL_FRAMES = 45      # ±1.8 s between golden kick and system pass frame
EDGE_FRAMES = 50     # system passes this close to a segment edge aren't FPs
                     # (play continues past the labeled window)


def load_golden(slug: str) -> list[dict]:
    out = []
    for p in sorted((Config.PROJECT_ROOT / "data" / "golden_events").glob(f"{slug}_*.json")):
        out.append(json.loads(p.read_text()))
    if not out:
        raise FileNotFoundError(f"no golden files for {slug}")
    return out


def score_passes(events, golden_docs) -> dict:
    sys_passes = [e for e in events if e.type == "Pass"]
    results = {"tp": [], "fn": [], "fp": [], "n_golden": 0}

    for doc in golden_docs:
        for seg in doc["segments"]:
            f0, f1 = seg["f_start"], seg["f_end"]
            seg_sys = [e for e in sys_passes if f0 <= e.frame <= f1]
            matched_sys = set()
            for g in seg["passes"]:
                results["n_golden"] += 1
                found = None
                for i, e in enumerate(seg_sys):
                    if i in matched_sys:
                        continue
                    if abs(e.frame - g["kick"]) > TOL_FRAMES:
                        continue
                    if e.player != g["from"]:
                        continue
                    if g["outcome"] == "complete" and g["to"] is not None \
                            and e.details.get("recipient") != g["to"]:
                        continue
                    found = i
                    break
                if found is not None:
                    matched_sys.add(found)
                    results["tp"].append((seg["name"], g))
                else:
                    results["fn"].append((seg["name"], g))
            unknown = seg.get("unknown_spans", [])
            for i, e in enumerate(seg_sys):
                if i in matched_sys:
                    continue
                if e.frame > f1 - EDGE_FRAMES or e.frame < f0 + EDGE_FRAMES:
                    continue
                if any(u["f0"] <= e.frame <= u["f1"] for u in unknown):
                    # Inside a span where the labeler could not see the ball
                    # (aerial/scramble) — the 25fps system may legitimately
                    # know more; count separately, neither TP nor FP.
                    results.setdefault("unverifiable", []).append((seg["name"], e))
                    continue
                results["fp"].append((seg["name"], e))
    return results


def score_carrier(gs, golden_docs, ball=None) -> dict:
    """Frame-level carrier accuracy on labeled control intervals."""
    from .events import ball_series, carrier_per_frame
    b = ball if ball is not None else ball_series(gs)
    carrier = carrier_per_frame(gs, b)
    tot = hit = wrong = missing = 0
    for doc in golden_docs:
        for seg in doc["segments"]:
            for c in seg["control"]:
                for f in range(c["f0"], c["f1"] + 1):
                    tot += 1
                    got = carrier.get(f)
                    if got is None:
                        missing += 1
                    elif got[0] == c["tid"]:
                        hit += 1
                    else:
                        wrong += 1
    return {"frames": tot, "correct": hit, "wrong": wrong, "unassigned": missing,
            "acc_when_assigned": round(hit / max(hit + wrong, 1), 3),
            "coverage": round((hit + wrong) / max(tot, 1), 3)}


def main():
    ap = argparse.ArgumentParser(description="Score events against the golden set")
    ap.add_argument("--match", default="sut-mla")
    args = ap.parse_args()

    gs = GameState.load(args.match)
    golden = load_golden(args.match)
    events, _ = detect_events(gs)

    r = score_passes(events, golden)
    tp, fp, fn = len(r["tp"]), len(r["fp"]), len(r["fn"])
    unv = len(r.get("unverifiable", []))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"golden passes: {r['n_golden']} | TP {tp}  FN {fn}  FP {fp} "
          f"(+{unv} unverifiable in blind spans) | "
          f"precision {prec:.2f}  recall {rec:.2f}")
    for name, g in r["fn"]:
        print(f"  FN {name}: kick f{g['kick']} {g['from']}->{g.get('to')} "
              f"({g['outcome']}) {g.get('note','')}")
    for name, e in r["fp"]:
        print(f"  FP {name}: f{e.frame} {e.player}->"
              f"{e.details.get('recipient')} ({e.details.get('outcome')})")

    c = score_carrier(gs, golden)
    print(f"carrier: coverage {c['coverage']*100:.0f}% of labeled control frames, "
          f"accuracy-when-assigned {c['acc_when_assigned']*100:.0f}% "
          f"(correct {c['correct']} wrong {c['wrong']} unassigned {c['unassigned']})")


if __name__ == "__main__":
    main()
