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

8. **Complete sparse-event survey vs SofaScore.** Per-type verdict (aggregate
   count ours-vs-Sofa):
   - **Clearances** (long def-third kick lost): **80% magnitude, per-team
     Spearman 0.67** — the one promising lead (right ballpark like passes,
     coverage-limited). Borderline-shippable "estimated"; deferred until the
     homography night fix. Not wired in.
   - **Dribbles** (carry past a near opponent): 1.4-3x over, inconsistent
     (sut-mla 86 vs 31). A take-on is a judged 1v1, not geometry. Marginal.
   - **Interceptions** 6-7x over (179 vs 27); **duels/recoveries** definition
     mismatch; **aerials** (no ball height), **fouls** (no contact model),
     **offside/blocks/miscontrol** (precise line / very low count) — not
     derivable.
   - **Rule**: an event is derivable iff it reduces to {ball_xy, player_xy,
     possession} on WIDE frames. Contact / ball-height / 1v1-intent are not.
   Reinforces the team/zone-level product boundary; clearances are the single
   sparse type worth revisiting after the coverage fix.

## The through-line

Every measurement points the same way: **team/zone-level analytics are close to
production quality (4.2pp possession vs the world reference); player-level is
capped by track fragmentation** and needs an upstream fix (re-ID tracker), not
more tuning. The two ranked levers are homography coverage (night games) and
track re-ID (player tier).

9. **Shot detection — definitively exhausted the geometric options.** Five
   approaches measured per-shot vs SofaScore (103 shots, 6 matches): ball-
   toward-goal 13-22%, camera-cut raw 86% recall / 654 cand/match (3% prec),
   +restart-after 39%, +box-occupancy 76% / 286 cand (6% prec), logistic
   regression on all signals (leave-one-match-out) — poor, ball features ~0
   weight. Blocker is structural: shots happen on the close-up where the ball
   isn't tracked, and ~650 cuts/match make context too generic. **The right
   tool is an action-spotting CNN on raw frames (SoccerNet approach) — and we
   already have the training labels (SofaScore shot times + validated clock
   map).** Or an image-space ball tracker. Goals already certain via oracle
   (7/7); the gap is non-goal shots. Documented in `shot_candidates.py`.

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

## Follow-up: shot action-spotting BUILT (src/shot_spotting.py)

Built the SoccerNet-style model the shot survey recommended, and it works
at a useful over-detect operating point:
- **Candidates** = wide→closeup camera cuts (86% shot recall alone).
- **Features** = ResNet18 embeddings of frames around each cut (wide build-up
  ++ closeup aftermath) + appearance-invariant geometric block. PCA-64 on the
  appearance block fixes a severe day/night overfit (train AUC 1.0 → test
  0.5-0.78 without it). Balanced LR (GB ties).
- **LOMO CV (103 shots, 6 matches): 89% of reachable / 77% of ALL shots at
  250 auto-ranked candidates/match** (reachable ceiling 86%; 2.6× fewer than
  650 raw cuts). Ranks real shots high (sut-mla 9:43→0.75 = SofaScore 9:39).
- With the goal oracle (100% of goals) this is the shot layer for the
  over-detect + human-QC workflow. Labels from SofaScore times + validated
  clock map. Next: complementary candidate source for the unreachable 14%,
  motion features, more labels.
