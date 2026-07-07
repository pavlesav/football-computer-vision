# Capability Assessment — 1.CFL Event-Data Pipeline

*Measured against SofaScore last-season data as ground truth. Written 2026-07-07.*

This is the honest, numbers-first statement of what the system can and cannot
do today, so product and business decisions rest on evidence, not hope. Every
number here is reproducible: `python -m src.sofa_eval` (team level) and
`--players` (player level).

## TL;DR

| Layer | Status | Evidence |
|---|---|---|
| Final score & goals | **Production** | Oracle 7/7 exact vs real results |
| Team possession % | **Production** | Mean error **4.2pp** vs SofaScore (1pp on well-covered games) |
| Which team dominated (pass share) | **Production** | Mean error **3.9pp** |
| Team pass volume | **Good** | **88% of SofaScore's count aggregate** (after the 2026-07-07 trust-threshold fix, up from 74%); night games still coverage-limited |
| Pass maps / territory / momentum (team & zone) | **Usable** | Built on the above; coverage-limited, not logic-limited |
| Per-player stats (who passed how much) | **NOT ready** | Rank corr **0.48**, 19% pass recall vs SofaScore |
| Shots (event-level) | **Not automated** | Detector conservative; needs candidate+human-tag workflow |
| Duels / fouls / aerials (event-level) | **Not built** | Genuinely manual across the whole industry |

The one-sentence version: **team-level and zone-level analytics are close to
production quality; player-level attribution is not, and is capped by track
fragmentation.**

## What's genuinely good (and why)

**Possession and pass-dominance are within ~4 percentage points of the world
reference.** Across all 7 test matches, mean possession error is 4.2pp, and on
well-covered games (sut-mla, jez-jed) it is 1pp. These are *coverage-invariant*
signals — they measure the split, not the absolute count — so they hold even
when we only see part of the match. This is the strongest evidence that the
possession→event logic is sound.

**Absolute pass volume tracks SofaScore closely.** After the 2026-07-07
trust-threshold fix (ceiling 0.75→0.35, validated against golden + SofaScore),
aggregate pass recall is **88%** (was 74%). The old adaptive gate pinned
well-lit matches to its 0.75 ceiling and discarded ~20pp of real passes; the
hand-labeled golden set confirmed the newly-admitted passes are genuine
(sut-mla p2 golden P/R improved 0.86/0.75 → 0.88/0.88). Night games remain
coverage-limited (~64%). The next lever for absolute recall is **homography
coverage on night games (PnLCalib fine-tune), not the event logic** — an
event-parameter sweep moved possession error ≤0.2pp (well-tuned already).

**Goals are certain.** The scoreboard oracle reads the broadcast graphic and
has matched the real final score on all 7 matches, including catching a
transient wrong scoreline (dec-mla 3-3→3-2) via the persistence check.

## What's not ready (and exactly why)

**Per-player statistics are unreliable today.** For players we successfully
identify by shirt number, our pass counts correlate with SofaScore at only
rho=0.48 and capture 19% of their true passes — far below the 97% team-level
recall. The cause is not the event logic; it is **identity attribution**. Only
~15-28% of detected events are attributed to an identified player, because:

- BoT-SORT produces 190-370 distinct track fragments *per half* that touch the
  ball (~1.7 events each) — every camera cut ends a track and starts new ones.
- A human can label ~40 tracks per half (the prominent ones), covering only
  24-36% of event participations.
- **Identity propagation** (this session) spreads those labels to adjacent
  fragments at perfect precision (CV 1.00) but low yield (+2-6pp attribution),
  because it cannot cross a camera cut — position continuity is meaningless
  once the camera jumps, and the persisted appearance embeddings were
  previously measured to carry no within-team identity signal.

So player-level quality is **capped by fragmentation**, and the fix is upstream
(a re-ID-capable tracker or a purpose-trained appearance model), not more
human labeling or more event tuning.

## The two levers that matter, ranked

1. **Homography coverage** — lifts team-level recall *and* every downstream
   number, especially on the ~50% of fixtures played under floodlights (night
   coverage collapses to ~17%). Fix: fine-tune PnLCalib on 1.CFL frames using
   the manual-calibration widgets already built. Multi-day, high payoff.
2. **Track fragmentation / re-ID** — the sole blocker on player-level stats.
   Fix: a tracker with appearance re-identification, or a trained ReID head, so
   a player survives camera cuts as one identity. Multi-week, unlocks the
   player-level product tier.

Everything else (shot candidate detection, set-piece tagging, report polish) is
comparatively small and well understood.

## Reproduce

```
python -m src.sofa_eval                 # team-level scorecard, all matches
python -m src.sofa_eval --players       # per-player pass validation
python -m src.identity_propagation --match sut-mla --half 1 --eval   # propagation CV
```
