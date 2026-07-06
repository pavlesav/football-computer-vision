# Overnight session log — 2026-07-07

Autonomous session. Goal set by Pavle: use SofaScore last-season data as
ground truth, iterate to make the system "somewhat decently" work, then plan
toward a business. Worked unattended.

## What got built / measured

1. **SofaScore ground-truth harness** (`src/sofa_eval.py`, `data/sofa_truth/`).
   Copied last-season stats for all 7 processed fixtures from the Polutka_Lab
   scrape (internal validation only, gitignored). Compares pipeline vs
   SofaScore: possession, pass counts, pass split, shots.
   - **Baseline: mean possession error 4.2pp, pass-split 3.9pp** (both
     coverage-invariant). Absolute pass recall ∝ homography coverage
     (sut-mla 56% cov → 97% of SofaScore pass count).

2. **Identity propagation** (`src/identity_propagation.py`). Mutual-best-match
   kinematic handoffs + jersey matches spread human anchors to fragments.
   - **CV precision 1.00**, invariant to thresholds (mutual-best is binding).
   - Honest yield: **low** (+2-6pp event attribution) — camera cuts fragment
     the event-bearing tracks and kinematics can't cross a cut.
   - Integrated into export via `events._resolve_idmap` (safe superset).
   - `HandoffGraph` builds links once (O(n·w)); `spread()` cheap per seed set.

3. **Encoding bug fixed** (cp1252 → UTF-8) across `identity.py`, `events.py`,
   `jersey_ocr.py`. Was crashing / mojibake-ing Montenegrin names. Reports now
   render clean names (verified on sut-mla report PNG).

4. **Per-player validation** (`sofa_eval --players`). For identified players:
   **rank corr 0.48, 19% pass recall** vs SofaScore. Player-level stats are
   NOT production-ready; team-level is. This is the key strategic boundary.

5. **Docs**: `CAPABILITY_ASSESSMENT.md` (measured state), `BUSINESS_PLAN.md`
   (grounded plan). All 6 labeled matches rebuilt with clean names +
   propagation.

## The through-line

Every measurement points the same way: **team/zone-level analytics are close to
production quality (4.2pp possession vs the world reference); player-level is
capped by track fragmentation** and needs an upstream fix (re-ID tracker), not
more tuning. The two ranked levers are homography coverage (night games) and
track re-ID (player tier).

## Next session (prioritized)

1. **Inference speed** (business-blocking for 48h delivery): `pnl_stride` 4-5 +
   FP16/TensorRT export → ~5-6h/match, all 5 in ~28h.
2. **PnLCalib night fine-tune** — biggest quality lever; ~half of fixtures are
   floodlit and collapse to ~17% coverage.
3. **Re-ID tracker** — the only path to production player-level stats.
4. **Shot candidate detector** — designed but not built; ball-toward-goal +
   dead-ball → candidate clips for human tag; validate recall against
   SofaScore `match_shots.timeSeconds` (needs continuous→per-half clock map).
5. **One-command pipeline + flagship opposition report** for a customer convo.

## Commits this session

- SofaScore ground-truth validation harness
- Identity propagation + per-player validation + encoding fix
- docs: capability assessment + business plan
