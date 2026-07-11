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
| Per-player stats (who passed how much) | **Developing** | Attribution **60%** of passes to a numbered player, **82%** of SofaScore XIs identified, recall **54%**, rho 0.38 — fully automatic (VLM numbers + ReID spread; see 2026-07-11 updates) |
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

So player-level quality was **capped by fragmentation**, and the fix is upstream
(a re-ID-capable tracker or a purpose-trained appearance model), not more
human labeling or more event tuning.

### Update 2026-07-11 — ReID cross-cut identity (the fragmentation fix, in part)

The "purpose-trained appearance model" above was built and measured, and it
works. A pretrained **osnet_ain_x1_0** ReID model (torchreid multi-source
domain-generalization weights) run over the stored player bboxes carries strong
within-team identity signal — **P(same-player pair closer than cross-player
pair) = 0.835** pooled across 12 labeled halves, vs **0.43 (chance)** for the
old ResNet18 team-classifier embeddings. The earlier "embeddings carry no
identity signal" result was specific to ResNet18; it does not generalize.

That unblocked a **cross-cut identity link** in `identity_propagation`: a
player's fragments on opposite sides of a camera cut — which kinematic handoff
structurally cannot bridge — are now joined by ReID similarity (same team, no
temporal overlap, mutual top-k above a cosine floor, jersey-conflict veto), so
the existing human shirt-number labels spread across cuts **with no new OCR or
human time**. Measured end-to-end against SofaScore, pooled over all 7 truth
matches (125 identified players):

| Metric | Before (kinematic only) | After (ReID links) |
|---|---|---|
| Per-player pass rank corr rho | 0.360 | **0.433** |
| Per-player pass recall | 16.6% | **43.1%** |
| Event-attribution % (passes → a numbered player) | 14.8% | **38.2%** |
| Team possession / pass-split error | 4.2 / 4.0 pp | **4.2 / 4.0 pp** (unchanged) |

Recall and attribution **2.6×**; rho **up** (more coverage stabilizes the
ranking). Team-tier is untouched because ReID links are same-team only, so a
mis-link moves a pass between teammates, never across teams; the pass/carrier
golden set is identity-invariant and also unchanged (0.79 / 0.94). Player-level
is now **developing**, not blocked — still short of production (43% recall), but
no longer capped. Reproduce: `python -m src.reid --eval` (the signal gate) and
`python -m src.sofa_eval --players` (the lift).

### Update 2026-07-11 (second pass) — fully-automatic identity at scale

Three additions turned the numbered-identity layer from human-seeded to
**automatic**, measured stage by stage (pooled over the 7 truth matches;
"XI" = SofaScore lineup players we identify at all):

| Stage | rho | pass recall | attribution | XI coverage |
|---|---|---|---|---|
| Baseline (ReID links, human seeds only) | 0.433 | 43.1% | 38.2% | 125/210 |
| + jersey-meta auto-anchors (easyocr) | 0.423 | 48.1% | 43.9% | ~130/210 |
| + VLM reader on sut-pet, demote-keep, lineup filter | 0.397 | 51.4% | 52.4% | 145/210 |
| + VLM reader on ALL matches | **0.381** | **54.2%** | **60.5%** | **173/210 (82%)** |

1. **Jersey-meta auto-anchors** (`identity_propagation._jersey_meta_anchors`):
   confident jersey metas seed propagation even with NO human identity file.
   The proof match is sut-pet (zero human labels): **attribution 5%→53%,
   rho −0.25→+0.40, XI 24/29** — the fully-automatic path works end-to-end.
2. **Qwen2-VL reader at scale** (`src/jersey_vlm.py`): 17 ms/crop on the RTX
   5070, ~10 min per half at the full 22k-crop budget — easyocr took ~an hour
   for 2-6× fewer confident numbers (dec-mla p1: 13 easyocr → 138 VLM metas).
3. **Uniqueness demotion inverted into a reunification signal**
   (`jersey_ocr.resolve_uniqueness`): shirt numbers are unique per team, so
   several gate-passed metas claiming one (team, number) are the same
   fragmented player. Demoted records are now kept (`demoted_for`) and seed
   propagation with the same key — **only for VLM files**: with easyocr's
   sparse reads this measurably reverses (sut-mla rho 0.42→0.04, structured
   digit confusions), so the seeds are reader-gated. A demoted-seed vote
   floor of 5 was swept and lost on every pooled metric — 3 stays.
   Confident (team, number) claims not in the official lineup
   (`data/lineups/{slug}.json`) are filtered as certain misreads.

**Measured negative result — do not lower the crop-height floor.** Running
the VLM with `min_box_h` 70 instead of 90 on sut-pet regressed everything
(rho 0.40→0.12, attribution 53%→49%): sub-90px torso crops read worse AND
displace tall crops from the capped 22k budget. The floor stays at 90.

Rank correlation dipped 0.43→0.38 while the matched-player set grew 125→173 —
newly-reached players are the thinly-observed ones, which drag rank stability
down even as every volume metric rises. Per-match rho now spans 0.70
(dec-mla) to 0.07 (mla-bud-2 — the night match whose team classifier is the
known weak link). The residual blockers for player-tier production are
homography coverage on night games and team-classification quality there —
identity attribution is no longer the cap.

### Update 2026-07-11 (third pass) — unsupervised team assignment validated (SOTA pattern)

Both the SoccerNet-2025 GSR winner (GSR-1) and the prior SOTA ("From
Broadcast to Minimap", CVPRW 2025) assign teams by *clustering appearance-ReID
embeddings* — no per-match supervision. Tested on our footage
(`src/team_cluster.py`, cosine k-means over the cached OSNet reid.npz + a
seed-based cluster→team grouping):

- **Agreement with the human-labeled classifier: 96.3% pooled, 94.1–97.7% on
  every one of 14 halves** — including the night match (mla-bud-2, 97.4%),
  where raw k-means clusters are still ~98% team-pure even though the two
  teams' centroids converge to 0.79–0.94 cosine under floodlights.
- Two grouping failure modes were found and engineered around: a team's kit
  splitting into multiple appearance clusters (sut-pet p2 — naive
  "two biggest clusters = two teams" paired two fragments of the SAME team),
  and night-game team-vs-team similarity exceeding ref-vs-team similarity
  (which collapses agglomerative merging). The seed rule (biggest cluster =
  team A; team B = biggest cluster below 0.90 cosine to A; join floor 0.80,
  else refs/other) handles both.
- **Conservative repair applied** (`--apply` flips a stored track team only on
  confident disagreement, margin ≥ 0.05, GKs excluded; parquet backed up):
  46 tracks across mla-bud-2 + jed-ars → rho 0.07→0.11 and 0.23→0.26, pooled
  rho 0.381→**0.392**, team pass-split error 4.0→3.9pp, nothing regressed.

**Operational consequence:** the ~15 min/match human team-labeling step is no
longer load-bearing — for new matches, teams can be bootstrapped from OSNet
clustering (embed all substantial tracks with `reid --all_tracks`, cluster,
write team_id) with the review UI as QC instead of source. This removes the
last per-match human step before the identity review itself.

## The two levers that matter, ranked

1. **Homography coverage** — lifts team-level recall *and* every downstream
   number, especially on the ~50% of fixtures played under floodlights (night
   coverage collapses to ~17%). Fix: fine-tune PnLCalib on 1.CFL frames using
   the manual-calibration widgets already built. Multi-day, high payoff.
2. **Track fragmentation / re-ID** — was the sole blocker on player-level stats;
   **partially addressed 2026-07-11** by the ReID cross-cut link above (recall
   17→43%). Remaining headroom: it still spreads only *human-seeded* numbers, so
   coverage is bounded by how many players were labeled — the next multiplier is
   **automatic numbers** (VLM jersey OCR) and eventually a ReID-native tracker
   so a player is one identity from detection, not reconstructed in post.

Everything else (shot candidate detection, set-piece tagging, report polish) is
comparatively small and well understood.

## Reproduce

```
python -m src.sofa_eval                 # team-level scorecard, all matches
python -m src.sofa_eval --players       # per-player pass validation
python -m src.identity_propagation --match sut-mla --half 1 --eval   # propagation CV
```
