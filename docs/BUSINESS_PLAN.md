# Business Plan — 1.CFL Match Analytics (working draft)

*Draft 2026-07-07. Grounded in the measured [Capability Assessment](CAPABILITY_ASSESSMENT.md).
This is a starting point for Pavle to edit, not a finished plan.*

## The opportunity

Montenegro's 1.CFL (and comparable small leagues) get **no professional event
data**. StatsBomb/Opta/Wyscout cover the top ~40 leagues because manual
annotation (2-3 analyst-hours per match) only pays off at top-flight media/
betting prices. That leaves every club below the top tier with, at best,
SofaScore's free aggregate stats — no pitch locations, no pass networks, no
territory, no opposition-specific detail, and posted days late.

**The wedge:** a broadcast-video pipeline that produces team- and zone-level
analytics for a fraction of the manual cost, delivered inside 48 hours of
kickoff. The defensible asset is data *nobody else has for this league at any
price* — pass locations, possession sequences, territory, momentum — not a
better version of something that exists.

## Who buys, and what they buy

**Primary: 1.CFL clubs (10 clubs).** A technical director / analyst who today
watches the match twice with a notebook. They buy:

- **Opposition reports** — how the next opponent builds up, where they
  concede territory, set-piece patterns, key distributors. This is the
  highest-value, most recurring product.
- **Own-team reports** — possession/territory trends, momentum swings, pass
  networks by zone, physical output (distance/speed from tracking).

**Secondary later:** the federation (league-wide data product, historical
archive), agents/players (individual highlight + stat packages), regional media
(graphics). Not v1.

## What we can sell *today* vs *soon* (from measurements)

| Product element | Sellable now? | Basis |
|---|---|---|
| Final score, goals, timeline | Yes | Oracle 7/7 exact |
| Possession %, territory, pass share | Yes | 4.2pp / 3.9pp vs SofaScore |
| Team pass networks **by zone** | Yes | Team-level pass geometry validated |
| Momentum / xT-style territory swings | Yes | Built on possession + locations |
| Physical data (distance, speed, heatmaps) | Yes | From tracking, camera-independent |
| Set-piece counts & locations | Mostly | Restart logic works; over-counts throw-ins |
| Shots + **own xG model** | Soon | Candidate-detect + human tag; xG on StatsBomb open data |
| **Per-player** passing/defensive stats | Not yet | rho 0.48, 19% recall — needs re-ID |

**Strategic consequence:** lead with **team & opposition analysis**, which is
production-grade and is exactly what a coach wants for the next fixture.
Position per-player stats as "coming", not core, until the tracker fix lands.
This is an honest and strong pitch — the opposition report doesn't *need*
per-player passing accuracy to be valuable.

## Legal / data posture (get a lawyer before selling)

- **SofaScore data is for internal validation only** — never resold or shown in
  a deliverable. It is their database right. (We use it here purely as a
  measuring stick.)
- **Facts are free**: scores, line-ups, fixtures are not anyone's property; take
  them from official league/club sources, not SofaScore.
- **Broadcast video**: confirm rights to process the federation/broadcaster
  feed for a derived analytics product. A partnership with the federation or
  broadcaster likely solves both the rights and the distribution question at
  once — and is the single highest-leverage business move.
- **Our xG is our own** — trained on StatsBomb *open* data (permissively
  licensed for research/commercial per their terms; verify) applied to our own
  shot features. This is a product asset, not a liability.

## Operations — can we hit 48h for a full gameweek?

5 matches per gameweek. Per match today: ~11h GPU (perception both halves) +
~15 min human identity labeling + report generation. Serial, that is ~2.3 days
of GPU on one RTX 5070 — **too slow**. Required before launch:

- **Inference speed**: `pnl_stride` 4-5 (validated) + FP16/TensorRT export of
  the HRNet + YOLO models (~2x, no quality loss) → ~5-6h/match → all 5 within
  ~28h, human loop in parallel. This is a launch-blocking engineering task, now
  a business requirement not a nicety.
- **Pipeline automation**: drop 5 videos → batch runs → oracle auto-checks the
  score → review pages generated → human labels ~15 min each → reports ship.
  Most of this exists (`run_batch`, `review_ui`, `sofa_eval` as a private QA
  gate); needs wiring into one hands-off flow.

## Rough economics (illustrative — validate with real conversations)

- **Cost per match**: GPU electricity + ~30-60 min human QC. Marginal cost is
  tens of euros, not hundreds. The asset is time-front-loaded (the pipeline).
- **Pricing hypotheses to test**: per-opposition-report, or a per-club season
  subscription (all their matches + all opponents). A single analyst salary
  saved, or one better signing/avoided loss, dwarfs a season subscription —
  that's the value anchor to sell against, not "cost of annotation".
- **Unit-economics reality**: with one GPU, ~5-10 matches/week is the ceiling;
  a second GPU doubles it. This is a boutique-margin, relationship-driven
  business at 1.CFL scale — the bigger prize is proving it on one league and
  templating to the dozens of comparable leagues worldwide.

## The real moat

Not the CV models (those commoditize). The moat is:

1. **League-specific tuning + the labeled asset** that accumulates every
   gameweek (your own growing ground-truth, your own xG, your own re-ID model).
2. **Distribution**: Pavle is a former 1.CFL player. That network — direct
   lines to technical directors and the federation — is something a foreign
   data company cannot replicate, and it is why *this* league is winnable.

## Milestones to a paid pilot

1. **Sharpen the deliverable** — one flagship opposition report a real 1.CFL
   analyst would pay for, from an already-processed match. Get a coach's
   reaction. (No new tech; uses today's team-level data.)
2. **Speed to 48h** — stride + TensorRT + one-command pipeline.
3. **Night-game coverage** — PnLCalib fine-tune (biggest quality lever).
4. **Player tier** — re-ID tracker; unlocks per-player upsell.
5. **One paid pilot club**, then templatize.

The technically hard, defensible half — dense spatial data from broadcast — is
already built and measured. The remaining work is coverage, speed, identity,
and a customer conversation.
