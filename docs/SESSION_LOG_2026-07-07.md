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

6. **Shot-candidate detection explored** (`src/shot_candidates.py`) — measured
   two signals vs SofaScore shot times (clock-aligned, verified via the goal
   anchor: oracle p2 10:45 ≈ SofaScore 10:40):
   - Ball-toward-goal: **recall 13-22%** — shots happen on close-ups where
     the ball isn't tracked (the goal itself gave zero ball candidates).
   - **Camera wide→close-up transition: recall ~83%** (9/9, 13/18, 25/30) —
     the director zooms on every shot. But ~500 transitions/match (≈2%
     precision); final-third ball filter halves recall.
   - **Finding**: camera-cut is the right high-recall foundation; needs a
     ball-independent precision filter (player-box-occupancy before the cut +
     restart type after). Not shipped; documented for next iteration.

7. **Homography trust threshold — VALIDATED WIN.** Swept `conf_min` against
   SofaScore (pass recall + possession guardrail) and golden. The adaptive
   ceiling of 0.75 was too conservative (pinned 4/6 matches, discarded ~20pp
   of passes). Lowered ceiling to **0.35** (`game_state.HOMOG_CONF_ABS_MAX`):
   - SofaScore aggregate pass recall **74% → 88%**; possession error flat
     (4.2pp); golden sut-mla p2 **improved** 0.86/0.75 → 0.88/0.88.
   - Chose 0.35 over 0.25 for safety (golden identical, less over-detection).
   - Event-parameter sweep (possession radius, carrier ball-speed): ≤0.2pp,
     within noise — event logic already well-tuned; leverage was the threshold.

8. **Sparse events (interceptions/duels/recoveries/fouls) — measured negative.**
   SofaScore counts are human-semantic; our geometric proxies mismatch 6-7x
   (interceptions 179 vs 27). Definitions don't map, and these events happen
   where tracking is weakest (contested/airborne; no ball height, no contact
   model). Per-team directional correlation only moderate (Spearman 0.69).
   Conclusion: sparse events are "human-tag or don't offer" — reinforces the
   team/zone-level product boundary. Not built.

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
4. **Shot detector v2** — build the camera-cut (83% recall) + player-box-
   occupancy precision filter; the ball-signal and clock-mapping already exist
   in `src/shot_candidates.py`.
5. **One-command pipeline + flagship opposition report** for a customer convo.

## Commits this session

- SofaScore ground-truth validation harness
- Identity propagation + per-player validation + encoding fix
- docs: capability assessment + business plan
