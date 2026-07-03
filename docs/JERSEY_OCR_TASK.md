# TASK PROMPT — Jersey-Number OCR for player identity (copy this whole file into a fresh chat)

You are working in the repository `football-computer-vision` (Windows 11, working
directory is the repo root). Read `CLAUDE.md` first — it is current and accurate.
Trust this prompt over any older impressions. Work autonomously, commit at each
coherent milestone, and follow every guardrail below exactly.

## Context — where the project stands

This pipeline extracts StatsBomb-style event data from Montenegro 1.CFL broadcast
video. Perception (YOLO + BoT-SORT + team classifier + PnLCalib homography) runs once
per half and persists a per-frame game state under
`output/game_state/{slug}/p{1|2}/` (`players.parquet` has one row per
(frame, track_id) with the image bbox `x1,y1,x2,y2`, `team_id`, pitch coords).
Analysis (ball tracking, events, reports) reads only the artifact.

Facts you need:

- BoT-SORT track ids restart every half. Tracks fragment heavily (a player is many
  tracks). `src/identity.py: consolidate_tracks(gs)` / `meta_map(gs)` merge tracks
  into per-half **meta-tracks** by kinematics (conservative; a meta ≈ one player).
- **Cross-half player identity does not exist for outfield players.** It was
  attempted with the persisted team-classifier ResNet18 embeddings and measured to
  be IMPOSSIBLE with them: same-player track pairs cosine 0.855 vs
  different-player-same-team 0.876, P(same>cross)=0.43 (chance). Do not revisit
  that approach. Jersey numbers are the chosen path: **same team + same shirt
  number in p1 and p2 = same player**.
- Goalkeepers are already unified across halves via positional roles
  (`goalkeeper-t{N}`, id base 900000, in `src/events.py: resolve_player`).
- Existing artifacts you can use: `sut-mla` p1 + p2 (two full stabilized halves),
  `bok-jed` p1 (4-min window), `bud-sut` p1 and `dec-mla` p1 (4-min windows).
- `src/backfill_embeddings.py` shows the exact pattern for decoding stored bboxes
  from video efficiently (sequential `cap.grab()` between sampled frames, then
  `cap.read()`); copy that decode pattern.
- The event-detection logic is golden-measured (P 0.83 / R 0.89 over 27 labeled
  passes). **You must not change any event-detection behavior.** Your work is
  extraction + export-layer identity only.

## Goal

1. New module `src/jersey_ocr.py`: read shirt numbers from stored player bboxes,
   vote them up to meta-track level, write
   `output/game_state/{slug}/p{N}/jersey_numbers.json`:
   `{"meta_numbers": {meta_id: {"number": int, "votes": int, "reads": int,
   "agreement": float}}, "params": {...}}`.
2. Evaluation subcommand that prints accuracy against the hand-labeled ground
   truth below + the two structural checks (uniqueness, cross-half consistency).
3. Integrate into the StatsBomb export (`src/events.py: resolve_player`): metas
   with a confident number resolve to a **period-independent identity**
   `{"id": 800000 + team*1000 + number, "name": "#<number>", "jersey_number": N}`.
   Resolution priority order (keep existing branches): named identity file >
   goalkeeper (900000 base, unchanged) > jersey number > consolidated meta id.
4. Regenerate the sut-mla match export + report and confirm cross-half
   unification is visible (top passers named `#N` aggregating both halves).

## Ground truth (hand-labeled from video contact sheets — use as the accuracy gate)

These (track_id → shirt number) pairs were read by a human from the broadcast
pixels. Track ids are per-artifact. High confidence unless noted.

`sut-mla` **period 2** (`GameState.load("sut-mla", period=2)`):

| track_id | team (kit) | number | note |
|---|---|---|---|
| 42894 | 1 (blue) | 99 | |
| 43265 | yellow player until ~f146404 | 5 | track then swaps onto a blue player — a track that reads different numbers over time is an ID swap, which your per-track vote may surface; aggregate votes are at meta level |
| 43355 | 0 (yellow) | 5 | same physical player as 43265's yellow phase |
| 43486 | 0 (yellow) | 28 | |
| 43634 | 0 (yellow) | 19 | |
| 43672 | 0 (yellow) | 3 | |
| 42973 | 0 (yellow) | 3 | likely the same player as 43672 (different track) |
| 43203 | 0 (yellow) | 77 | medium confidence |

`bok-jed` **period 1** (`GameState.load("bok-jed")`):

| track_id | team (kit) | number | note |
|---|---|---|---|
| 192 | 0 (navy) | 9 | |
| 146 | 1 (white) | 8 | |
| 129 | 1 (white) | 30 | |
| 150 | 0 (navy) | 10 | |
| 315 | 0 (navy) | 14 | medium confidence |

Scoring rule: over the ground-truth tracks where your system assigns a number to
the track's meta, **assigned-number accuracy must be ≥ 90%**. Abstaining
(no confident number) is always acceptable; a wrong confident number is not.

## Plan — execute in this order

**Step 0 — read before writing.** `CLAUDE.md`, `src/backfill_embeddings.py`,
`src/identity.py` (`consolidate_tracks`, `meta_map`, `load_identity_map`),
`src/events.py` (`_sb_records_for_period.resolve_player`), and skim
`src/game_state.py` (`GameState.load(slug, period=...)`, `gs.dir`, `gs.players`).

**Step 1 — crop prototype + mandatory visual check.** Numbers are on the BACK of
shirts; a player facing the camera has no number, so most crops will read nothing —
that is expected and fine. Torso crop from a bbox `(x1,y1,x2,y2)` with
`h = y2-y1, w = x2-x1`: vertical `y1+0.12h .. y1+0.55h`, horizontal
`x1+0.15w .. x2-0.15w`. Only use boxes with `h >= 90` px. Upscale the crop 4x
(`cv2.resize`, cubic) before OCR. Before writing the batch pipeline: extract ~24
sample crops from large boxes of the sut-mla p2 ground-truth tracks above, save a
contact sheet PNG, and **look at it** — confirm digits are legible in at least
some crops and the crop geometry doesn't cut them. If digits are systematically
clipped, adjust the geometry constants and re-check. Do not skip this step.

**Step 2 — extraction module** (`src/jersey_ocr.py`, CLI:
`python -m src.jersey_ocr --match sut-mla --half 2`).

- OCR: `easyocr.Reader(["en"], gpu=False, verbose=False)`, single shared
  instance, `readtext(crop, allowlist="0123456789")`. Accept a read only if
  easyocr confidence ≥ 0.5 and the text is 1–2 digits parsing to 1..99.
- Track selection: only tracks with ≥ 25 rows in `players.parquet` (short
  flickers don't matter for events). For each selected track, pick up to 12
  frames with the tallest boxes (`h >= 90`), spread in time (sort by height,
  then de-duplicate frames closer than 25 frames apart).
- Budget: process metas in descending total-frames order; stop after
  `--max_crops` total crops (default 3000). Decode with the grab/retrieve
  pattern from `backfill_embeddings.py` — random seeking per frame is too slow.
- Voting: collect reads per track; map tracks to metas via
  `identity.meta_map(gs)`; a meta's number is the plurality read. Accept only if
  `votes >= 3` and `agreement = votes/reads >= 0.6`. Treat "2" and "28" as
  different numbers — never merge substrings.
- Also write per-track vote detail into the JSON (under `"track_reads"`) — a
  track whose reads split between two numbers is an ID-swap suspect; print the
  top 10 such tracks as a diagnostic (do not auto-split anything).

**Step 3 — evaluate** (`python -m src.jersey_ocr --match sut-mla --half 2 --eval`
and the same for bok-jed p1). Report:

- Accuracy vs the ground-truth table (through the track's meta assignment).
- **Uniqueness**: within (team, period), no two metas may share a number. Zero
  violations may be shipped — resolve collisions by keeping the higher-vote meta
  and demoting the other to no-number.
- **Cross-half consistency** (sut-mla): the number sets for team 0 in p1 vs p2
  should overlap substantially (≥ ~60% of the smaller set; substitutions reduce
  it). Same for team 1. If overlap is near-random, your reads are noise —
  tighten thresholds (votes, agreement, confidence, min height) rather than
  shipping. Print the matched pairs.

**Step 4 — tune only against those gates.** If accuracy < 90%: raise the easyocr
confidence floor, require `votes >= 4`, or raise the min box height. Coverage
(how many metas get numbers) is secondary — precision is the product.

**Step 5 — export integration** (`src/events.py`). In `_sb_records_for_period`,
load `jersey_numbers.json` for the period (pass it in like `meta_map` is passed;
wire it through `export_events`, `export_match_events`, and `main`). In
`resolve_player`, after the identity-file branch and the goalkeeper branch:
if the tid's meta has a confident number, return
`{"id": 800000 + team*1000 + number, "name": f"#{number}",
"jersey_number": number}`. Team comes from the meta table
(`consolidate_tracks` output) — pass it alongside. Numbers only unify across
halves via this shared id; do NOT touch `PERIOD_TID_OFFSET` logic for the
remaining unnumbered metas.

**Step 6 — regression proof.** Run all three golden evals and confirm the
numbers are IDENTICAL to the baseline (your change is export-layer only):

```
python -m src.golden_eval --match sut-mla --half 1   # P 0.79 R 0.94, carrier 62%/98%
python -m src.golden_eval --match sut-mla --half 2   # P 0.86 R 0.75, carrier 76%/99%
python -m src.golden_eval --match bud-sut            # P 1.00 R 1.00
```

Any deviation means you changed detection behavior — find and undo it.

**Step 7 — regenerate deliverables.**
`python -m src.run_match --match sut-mla --home_team 1` (it will skip perception/
stabilize/oracle and rebuild events + report). Open
`output/reports/sut-mla/sut-mla_match_report.png` and verify top passers show
`#N` entries that aggregate both halves (no `-h2` suffix on numbered players).

**Step 8 — docs + commits.** Update `CLAUDE.md` (module line in the structure
tree + a short subsection under the identity/roles material + roadmap item 4/5
status) and `README.md` (structure tree + status table row). Commit in small
steps: (a) extraction module + eval, (b) export integration + regression proof,
(c) docs. Plain commit messages.

## Guardrails — read twice

- **Never change event-detection logic** (`detect_events`, touches, kicks,
  dead-ball, ball tracker, roles, homography). Golden must stay bit-identical.
- **Do not launch GPU perception** (`src.pipeline`) — not needed for this task.
- **Do not use the ResNet embeddings for identity.** Measured dead end.
- Interpreter: `/c/Users/PC/AppData/Local/Python/bin/python` (bare `python` can
  hit a Windows Store stub). Windows console is cp1252 — ASCII only in prints.
- easyocr on CPU (`gpu=False`). ~0.3–0.8 s per crop; the 3000-crop default
  budget ≈ 20–40 min per half. Run long extractions with output redirected to a
  log file.
- Git: plain commit messages. **No Claude/Anthropic attribution, no co-author
  trailers, no "Generated with" footers** — repository rule.
- A wrong confident number is worse than no number. When a gate fails, tighten;
  never loosen a gate to make a report look better.
- If you find the ground-truth table disagrees with what you see in the pixels,
  investigate before overriding: regenerate the crop for that exact track and
  frame range and look. The table was read by a human from real frames; two of
  the rows are marked medium-confidence.

## Definition of done

1. `jersey_numbers.json` exists for sut-mla p1, sut-mla p2, bok-jed p1.
2. Eval prints: ground-truth accuracy ≥ 90% (on assigned), zero uniqueness
   violations, sut-mla cross-half overlap reported and plausible.
3. Golden evals bit-identical to the baselines above.
4. Merged sut-mla export + report show `#N` players unified across halves.
5. Docs updated; all work committed in small, plain-message commits.
