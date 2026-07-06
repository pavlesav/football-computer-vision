"""
Match-level validation against SofaScore ground truth.

SofaScore last-season data (copied to ``data/sofa_truth/{slug}/`` from the
Polutka_Lab scrape — INTERNAL VALIDATION ONLY, never redistributed) gives
per-match team aggregates: possession %, total/accurate passes, shots,
throw-ins, corners, xG, plus per-player pass counts. We compare our pipeline
output to these to get an objective "how decent are we" scorecard and to know
which layer to fix next.

Caveats that make some deltas expected, not bugs:
* Coverage. We only produce events on trusted-homography, wide-shot, gameplay
  frames. A match at 30% trusted coverage cannot show 100% of SofaScore's
  passes — so we report our pass count BOTH raw and scaled by trusted
  coverage, and treat *possession split* and *pass ratio between teams* (both
  coverage-invariant) as the primary fidelity signals.
* Definitions differ. SofaScore counts completed + attempted passes with a
  human definition; ours is a detector. Absolute counts will differ; the
  RELATIONSHIP (which team passed more, possession share, pass-length
  distribution) is what should match.

Run::

    python -m src.sofa_eval                 # scorecard over all matches
    python -m src.sofa_eval --match sut-mla  # one match, verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config

TRUTH_DIR = Config.PROJECT_ROOT / "data" / "sofa_truth"

# home/away in SofaScore == home/away on the broadcast graphic. Our classifier
# team ids are arbitrary; this maps the HOME side to a classifier id (same
# values verified for the goal oracle / run_batch).
HOME_CLASSIFIER_TEAM = {
    "jed-ars": 1, "jez-ars": 0, "jez-jed": 0, "sut-pet": 0,
    "mla-bud-2": 1, "dec-mla": 1, "sut-mla": 1,
}


def load_truth(slug: str) -> dict:
    """Flatten SofaScore statistics.json into {stat_name: (home, away)}."""
    p = TRUTH_DIR / slug / "statistics.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d["statistics"] if isinstance(d, dict) and "statistics" in d else d
    allp = next(x for x in s if x.get("period") == "ALL")
    out = {}
    for g in allp["groups"]:
        for it in g["statisticsItems"]:
            out[it["name"]] = (it.get("home"), it.get("away"))
    return out


def _num(v):
    """'219' -> 219; '25/57 (43%)' -> 25; '46%' -> 46.0."""
    if v is None:
        return None
    sv = str(v)
    if "/" in sv:
        return float(sv.split("/")[0])
    if sv.endswith("%"):
        return float(sv[:-1])
    try:
        return float(sv)
    except ValueError:
        return None


def pipeline_metrics(slug: str) -> dict:
    """Our numbers from the merged events JSON, mapped to home/away."""
    from .events import export_match_events  # noqa: F401 (ensure importable)
    p = Config.OUTPUT_EVENTS_DIR / f"{slug}_events.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    s = d["summary"]
    home_t = HOME_CLASSIFIER_TEAM[slug]
    away_t = 1 - home_t
    ht, at = str(home_t), str(away_t)

    def split(dct):
        return (dct.get(ht, 0), dct.get(at, 0))

    poss = s.get("possession_pct", {})
    passes = s.get("passes", {})
    comp = s.get("pass_completion_pct", {})
    return {
        "possession": split(poss),
        "passes": split(passes),
        "pass_completion": split(comp),
        "shots": split(s.get("shots", {})),
        "trusted_pct": s.get("homography_trusted_pct"),
        "n_events": s.get("n_events"),
    }


def _fmt_pair(p):
    if not p:
        return "   -  /   -  "
    a, b = p
    fa = f"{a:.0f}" if isinstance(a, (int, float)) else "-"
    fb = f"{b:.0f}" if isinstance(b, (int, float)) else "-"
    return f"{fa:>4} /{fb:>4}"


def scorecard(slugs: list) -> None:
    print(f"\n{'match':<11} {'metric':<14} {'SofaScore':>13}  "
          f"{'pipeline':>13}   note")
    print("-" * 76)
    agg = {"poss_err": [], "pass_ratio_err": []}
    for slug in slugs:
        if not (TRUTH_DIR / slug).exists():
            continue
        truth = load_truth(slug)
        pipe = pipeline_metrics(slug)
        if not pipe:
            print(f"{slug:<11} (no pipeline events yet)")
            continue
        tp = (_num(truth["Passes"][0]), _num(truth["Passes"][1]))
        ta = (_num(truth.get("Accurate passes", (None, None))[0]),
              _num(truth.get("Accurate passes", (None, None))[1]))
        tposs = (_num(truth["Ball possession"][0]),
                 _num(truth["Ball possession"][1]))
        tshots = (_num(truth["Total shots"][0]), _num(truth["Total shots"][1]))

        # coverage-invariant fidelity: possession split + pass ratio
        pposs = pipe["possession"]
        poss_err = abs(pposs[0] - tposs[0]) if tposs[0] is not None else None
        pr_truth = tp[0] / (tp[0] + tp[1]) * 100 if tp[0] and tp[1] else None
        pp = pipe["passes"]
        pr_pipe = pp[0] / (pp[0] + pp[1]) * 100 if (pp[0] + pp[1]) else None
        pr_err = (abs(pr_pipe - pr_truth)
                  if pr_truth is not None and pr_pipe is not None else None)
        if poss_err is not None:
            agg["poss_err"].append(poss_err)
        if pr_err is not None:
            agg["pass_ratio_err"].append(pr_err)

        tcov = pipe.get("trusted_pct")
        cov_note = f"cov {tcov:.0f}%" if tcov is not None else ""
        print(f"{slug:<11} {'possession %':<14} {_fmt_pair(tposs):>13}  "
              f"{_fmt_pair(pposs):>13}   Δ{poss_err:.0f}pp" if poss_err
              is not None else f"{slug:<11} possession")
        print(f"{'':<11} {'passes':<14} {_fmt_pair(tp):>13}  "
              f"{_fmt_pair(pp):>13}   {cov_note}")
        print(f"{'':<11} {'pass split %':<14} "
              f"{pr_truth:>5.0f} /{100-pr_truth:>4.0f}  "
              f"{pr_pipe:>5.0f} /{100-pr_pipe:>4.0f}   Δ{pr_err:.0f}pp"
              if pr_err is not None else "")
        print(f"{'':<11} {'shots':<14} {_fmt_pair(tshots):>13}  "
              f"{_fmt_pair(pipe['shots']):>13}   (we don't detect; oracle=goals)")
        print()

    if agg["poss_err"]:
        mp = sum(agg["poss_err"]) / len(agg["poss_err"])
        print(f"MEAN possession error: {mp:.1f} pp "
              f"(coverage-invariant fidelity signal)")
    if agg["pass_ratio_err"]:
        mr = sum(agg["pass_ratio_err"]) / len(agg["pass_ratio_err"])
        print(f"MEAN pass-split error:  {mr:.1f} pp "
              f"(which team dominated — coverage-invariant)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=None)
    args = ap.parse_args()
    slugs = ([args.match] if args.match
             else sorted(p.name for p in TRUTH_DIR.iterdir() if p.is_dir()))
    scorecard(slugs)


if __name__ == "__main__":
    main()
