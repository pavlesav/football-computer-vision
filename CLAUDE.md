# Football Computer Vision - Project Guide

## What This Project Does

Semi-automated event data extraction from broadcast football (soccer) footage from the Montenegro First Division (1.CFL). The pipeline takes a full-match video and produces:
- Player and ball detection (bounding boxes) with persistent tracking
- Team classification (Team A vs Team B) via jersey appearance
- Broadcast period detection (first half, second half kickoff frames)
- Pitch homography (pixel-to-pitch coordinate mapping)
- *(Planned)* Event data: passes, shots, possession changes

**End goal:** Produce match event data for every 1.CFL game. Not fully unsupervised — a human-in-the-loop workflow with 1-3 hours of labeling/correction per game is the target.

## Tech Stack

- **Python 3.10+** on Windows 11, NVIDIA RTX 5070 (sm_120, CUDA 12.8)
- **YOLOv8m** (Ultralytics) - fine-tuned for 2-class detection (person, ball)
- **BoT-SORT** - multi-object tracking via Ultralytics' built-in tracker
- **YOLOv8n-seg** - instance segmentation to mask player silhouettes (background removal)
- **ResNet18** (frozen ImageNet backbone) - 512-dim CNN embeddings from masked player crops
- **PCA + KNN** (scikit-learn) - dimensionality reduction (512->32) then KNN (k=5) for supervised team classification; GMM is still available for unsupervised exploration
- **easyocr** - match clock reading for period detection
- **PnLCalib** (HRNetV2-W48) - pretrained pitch keypoint + line detection for camera calibration (3x4 projection matrix)
- **OpenCV** - video I/O, image processing, homography estimation

## Project Structure

```
football-computer-vision/
├── src/                               # All Python source code
│   ├── __init__.py                    # Lazy imports for all modules
│   ├── config.py                      # All paths, thresholds, hyperparameters
│   ├── detection.py                   # PlayerDetector (YOLO), BallInterpolator
│   ├── segmentation.py                # YOLOv8-seg masks, ResNet18 CNN embeddings
│   ├── team_classifier.py             # TeamClassifier: PCA + KNN (k=5) supervised classification
│   ├── tracking.py                    # Tracker: track-level team assignment
│   ├── broadcast.py                   # Camera cut detection, period detection via clock OCR
│   ├── camera_motion.py               # Optical-flow tracker: propagates homography across PnLCalib dropouts
│   ├── manual_calibration.py          # Landmark / drag-line / line-adjust GT widgets + saved homography seeds
│   ├── homography.py                  # Classical pitch line detection (legacy, not primary)
│   ├── visualization.py               # Annotator: ellipses, triangles, bboxes
│   ├── video_utils.py                 # Video I/O, clip extraction (ffmpeg)
│   ├── dataset.py                     # Training data management, train/val split
│   ├── run_pnlcalib_video.py          # PnLCalib homography on video clips
│   ├── run_demo.py                    # Demo clip renderer: tracking + names + speed/distance + minimap (legacy showcase)
│   ├── game_state.py                  # Persisted per-frame game-state artifact (Parquet, per half) — the perception↔analysis boundary
│   ├── pipeline.py                    # PerceptionPipeline: runs perception once → output/game_state/{slug}/p{N}/ (partial writes every 15k frames)
│   ├── ball_tracker.py                # Pitch-space Kalman ball tracker over the game state (no GPU/video)
│   ├── events.py                      # Possession/kicks → touches → spells → per-half + merged match SB JSON (+ oracle goals, restarts/play_pattern)
│   ├── score_ocr.py                   # Scoreboard goal-oracle: OCR both league layouts (text-token + red-box) → certain goals, bracketed timing
│   ├── report.py                      # One-page match report (mplsoccer) from the merged SB JSON → output/reports/{slug}/
│   ├── run_match.py                   # Orchestrator: perception both halves → stabilize → oracle → events → report (resumable)
│   ├── backfill_embeddings.py         # Recompute embeddings.npz for artifacts predating its persistence (stored bboxes, no re-run)
│   ├── roles.py                       # Attack direction (defensive-shape vote) + GK identification (positional signature)
│   ├── identity.py                    # Track→meta-track consolidation + naming widget + data/identities/{slug}.json
│   ├── team_repair.py                 # Kit-hue audit of track team labels (flip clear errors, flag ID-swap suspects)
│   ├── stabilize.py                   # Offline homography smoothing (kills the 8Hz overlay jitter) + player-pos recompute
│   ├── golden_eval.py                 # Score detected passes/carrier vs hand-labeled golden set (precision/recall)
│   └── render_game_state.py           # Annotated review MP4 from the artifact (supports --start_sec/--duration_sec clips)
├── models/                            # All models and weights
│   ├── detection/                     # Fine-tuned YOLOv8m (weights + training artifacts)
│   │   ├── weights/best.pt           # Trained model weights
│   │   ├── args.yaml                 # Training config
│   │   └── results.csv               # Training metrics
│   ├── segmentation/
│   │   └── yolov8n-seg.pt             # Instance seg for team classification
│   ├── pitch_keypoints/
│   │   └── soccana_keypoints.pt       # Soccana YOLOv11 keypoint model (backup)
│   └── pnlcalib/                      # PnLCalib — primary homography model (gitignored, ~640MB)
│       ├── model/                     # HRNetV2 architecture code
│       ├── utils/                     # Calibration + heatmap utilities
│       ├── config/                    # HRNet YAML configs
│       └── weights/{SV_kp,SV_lines}   # Pretrained weights (~253MB each)
├── data/
│   ├── golden_events/                 # Hand-labeled ground truth (control intervals + passes) for golden_eval
│   ├── identities/                    # {slug}.json — meta-track → player name/number (written by the naming widget)
│   ├── period_detection_results.json  # Cached period boundaries for all 16 matches
│   ├── manual_calibration/            # {slug}_frame_{N:07d}.json — manual homography seeds for Tier-3 matches
│   └── object_detection/              # Labeled training data for YOLO fine-tuning
│       ├── data.yaml                  # YOLO dataset config (2 classes)
│       ├── images/{all,train,val}/    # 971 annotated frames
│       ├── labels/{all,train,val}/    # YOLO-format labels (class cx cy w h)
│       └── cvat_exports/              # Archived CVAT annotation exports
├── videos/                            # Source match videos (gitignored)
├── output/                            # All generated outputs (gitignored)
│   ├── classifiers/                   # Per-game {slug}_classifier.pkl + {slug}_labels.npz
│   ├── classifier_validation/         # Annotated 2-min clips for visual QC of team classification
│   ├── demo/                          # Demo MP4s (`{slug}_{ts}_demo.mp4`, `{slug}_{ts}_discover.mp4`)
│   ├── game_state/{slug}/p{1,2}/      # players/frames/ball parquet + embeddings.npz + meta.json per half
│   ├── events/                        # {slug}_events.json (match) + {slug}_p{N}_events.json + {slug}_goal_oracle.json
│   └── reports/{slug}/                # One-page match report PNG
├── notebooks/
│   ├── 01_team_classification.ipynb   # Per-game labeling widget → KNN classifier + labels.npz + review + validation
│   ├── 02_homography.ipynb            # Homography visualization + model comparison
│   ├── 03_event_detection.ipynb       # Load game state → detect_events → pass-map/possession QC → StatsBomb-lite export
│   ├── 04_demo_video.ipynb            # Demo render + label/calibrate/compare iteration loop
│   └── detection_training.ipynb       # YOLO training + error analysis
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Roadmap (in priority order)

1. ~~**Broadcast analysis**~~ — Done. Unsupervised period detection works on all 16 matches (15 PASS, 1 WARN).
2. **Pitch homography** — In progress. PnLCalib pretrained model + post-processing (v2), Lucas-Kanade optical-flow propagation (`camera_motion.py`), and manual seeds (`manual_calibration.py`) all wired into the runner. v2 averages 71% coverage across 16 matches. The demo notebook (`04_demo_video.ipynb`) added a label/calibrate/compare iteration loop with a line-adjust ground-truth widget seeded from the pipeline's own P. Next step: use that loop on a representative clip to diagnose the residual drift/wrong-overlay failures (PnLCalib output vs `refine_projection_to_lines` bias vs flow-propagated bad seed) and tighten whichever stage is responsible.
3. **Event detection** — Working at **full-match scale, golden-measured on 4 segments / 3 matches / 27 passes**. `src.pipeline` persists a per-frame **game state**; ball tracker + roles (attack direction, GKs) + rule-based events (incl. restart detection → throw-in/corner/goal-kick passes + SB play_pattern) derive a StatsBomb-v4-shaped stream. Golden scorecard: **combined P 0.83 / R 0.89** — sut-mla FH 0.79/0.94, sut-mla SH 0.86/0.75, bud-sut 1.00/1.00, carrier 98-99% when assigned (Tier-1 conditions); bok-jed Tier-2 baseline is 0/0 with named causes (homography-bounded coverage, crowd carrier misattribution, roles need ≥ a half of data). Remaining error classes are *named*: ID swaps / track team-contamination (the entire remaining sut-mla FN/FP set), restart over-detection (~3x true throw-in count; hurts play_pattern fidelity, not pass precision), non-goal shot outcomes.
4. **Cross-shot player matching** — First version working (`src/identity.py`): GK tracks merge via roles; outfield via team + kinematic continuity + a meta-level long-gap/short-distance pass. bud-sut: 320 tracks → 78 metas, top-40 = 88% of player-frames. Close-up-heavy halves fragment (sut-mla FH 27%). **Measured negative result (2026-07-03): the team-classifier ResNet18 embeddings carry NO within-team identity signal** (same-player track pairs cosine 0.855 vs different-player-same-team 0.876; P(same>cross)=0.43 = chance) — cross-half/appearance ReID must NOT use them. GKs get a stable cross-half identity via roles (`goalkeeper-t{N}` in exports); outfield cross-half unification happens through the naming workflow (name both halves identically) until jersey-number OCR or a purpose-trained ReID model lands.
5. **Jersey number / player identity** — Naming workflow exists: `build_identity_widget` (best-crop gallery per meta-track, ~10-15 min/match with public lineups) → `data/identities/{slug}.json` → export resolves `player: {id, name, jersey_number}` automatically.
6. ~~**"First complete match" milestone**~~ — **Done (2026-07-03), sut-mla is the first complete match**: per-half artifacts (`p1|p2`, legacy flat dirs still load), both halves processed + stabilized, one merged SB events JSON (1,434 events, 877 passes, possession 54/46, 207 possession changes, attack directions verified to flip at halftime), **scoreboard goal-oracle** (`src/score_ocr.py`) pixel-validated — final score OCR 1-0 == real result, the goal anchored within ~1s of the ball crossing the line — and the **first one-page match report** (`src/report.py`, mplsoccer). Key insight from QC: the goal happened on a close-up camera where events are correctly paused, so the oracle is the *only* honest source of goals.
7. **NEXT: batch Tier-1/2 matches** via `src/run_match.py` (one command per match: perception both halves → stabilize → oracle → events → report; resumable via `output/match_runs/{slug}_status.json`; launch detached — a harness-tied background run died at 90% once). The generalized score_ocr parses all six surveyed scoreboard layouts (both families: text-token score and red-box digits). Then the correction UI informed by real per-match error volumes; jersey-number OCR for cross-half outfield identity.

## Data Flow (01_team_classification.ipynb)

The per-game labeling workflow runs in two passes:

```
Pass 1: Detect + Track + Sample Embeddings
  For every frame (across 6 × 90s windows spanning both halves):
    YOLO detection + BoT-SORT tracking -> all_detections[]
  Every 15th frame:
    YOLOv8-seg (full frame) -> person masks
    Masked crops -> ResNet18 -> 512-dim embeddings
    Store per-track: track_embeddings[track_id].append(embedding)

Between passes: Label & Fit
  Top 200 tracks shown in widget -> human assigns A / B / Other
  Labeled embeddings -> PCA (512->32) fit -> KNN (k=5) fit

Pass 2 (validation): Detect + Classify + Annotate
  Re-read a 2-min clip, collect embeddings, classify tracks by majority vote
  Draw team-colored ellipses + ball triangles -> MP4 in output/classifier_validation/
```

**Key design choice:** Team classification is done at the **track level**, not per-frame. Each track casts one vote per sampled embedding; the majority label wins. This is more robust than a single prediction on the median embedding — a brief ID swap or handful of noisy frames gets outvoted by the dominant player's embeddings.

## Detection Model

- **Architecture:** YOLOv8m (25M params)
- **Classes:** 2 - `person` (0), `ball` (1)
- **Training:** 150 epochs, imgsz=1280, batch=8, mosaic=1.0, copy_paste=0.3, patience=30
- **Dataset:** 971 frames across 16 matches (822 train / 149 val, match-level split)
- **Best model:** `models/detection/weights/best.pt`
- **Metrics:** mAP50=0.944, mAP50-95=0.748, Precision=0.952, Recall=0.892
- **Weight resolution:** `Config.resolve_yolo_model()` resolves to `models/detection/weights/best.pt`

Roles like goalkeeper, referee, and team assignment are NOT part of detection — they're handled entirely in post-processing (team classifier + future position heuristics).

## Team Classification Pipeline

One-time per game (~15 min labeling), saved to `output/classifiers/` for reuse.

1. **Segmentation:** YOLOv8n-seg runs on the full frame (not crops!) to produce person silhouette masks
2. **Masking:** Each player's crop is masked — background pixels replaced with mean foreground color to avoid CNN bias
3. **Embedding:** Frozen ResNet18 backbone produces 512-dim vectors per masked crop, resized to 128x64 (tall format for full-body players)
4. **PCA:** 512 → 32 dims, removes noise
5. **Human labeling:** ~200 track crops shown in a widget; user assigns A/B/Other to each
6. **KNN fit (k=5):** Trained on labeled embeddings in PCA space. Uses local neighborhood voting rather than single centroid — handles referees/GKs that sit between team clusters better than nearest-centroid
7. **Track-level prediction:** Each track casts one vote per sampled embedding; majority label wins (more robust than single predict on median embedding — brief ID swaps or noisy frames get outvoted)

**Saved artifacts per game:**
- `output/classifiers/{slug}_classifier.pkl` — fitted PCA + KNN model
- `output/classifiers/{slug}_labels.npz` — raw labeled embeddings + integer labels (source of truth; allows `TeamClassifier.refit_knn()` without re-labeling). Also stores `review_track_ids` + `review_track_labels` (track-level, includes skipped=-1) for the review widget.
- `output/classifiers/{slug}_pass1.pkl` — raw track embeddings + best crops from Pass 1 (used by review and validation cells)

**Review widget** (`01_team_classification.ipynb`): loads labels from disk cross-session, shows all labeled crops grouped by team (A / B / Other / Skipped) so mislabeled outliers are visually obvious. "Save corrections" button re-saves the npz and refits the classifier in one click.

**Validation cell** (`01_team_classification.ipynb`): runs detection + classification on a 2-minute clip and saves an annotated video to `output/classifier_validation/{slug}_team_check.mp4`. Used to visually QC classification quality on the harder games (night/dusk, similar jerseys) before committing to full-game processing.

**`TeamClassifier` API:**
- `fit_supervised(embeddings, labels)` — fit PCA + KNN from labeled data
- `classify_tracks(track_embeddings)` — classify all tracks by majority vote over per-embedding predictions
- `save(path)` / `load(path)` — serialize/deserialize model state
- `refit_knn(pkl_path, labels_path, n_neighbors=5)` — refit KNN from saved npz without re-labeling

**Status:** 16/16 matches labeled. The unsupervised GMM path (`fit()`) remains available for exploration but is not used in production.

## Broadcast Analysis (`src/broadcast.py`)

Fully unsupervised segmentation of a full-match broadcast into game periods.

**Pipeline:**
1. `detect_cuts` — HSV histogram diff between consecutive frames, fixed threshold 0.4. Number of cuts is production-dependent (ars-dec: 4, jez-jed: 33, bud-sut: 391, bok-jed-2: 586). Few cuts is not a bug — some 1.CFL broadcasts are single-camera productions.
2. `build_segments` — Merges sub-3-frame flash transitions.
3. `check_scoreboard` — Otsu-thresholded Canny edge density over `SCOREBOARD_ROI` per segment; classifies gameplay vs replay/graphics.
4. `detect_period_starts` — OCRs the match clock (easyocr) on **60 uniformly-spaced frames across the whole broadcast** (NOT per-segment). Two extrapolation rules:
   - clock < 40:00 -> `first_half_start = broadcast_time - clock_time`
   - clock >= 50:00 -> `second_half_start = broadcast_time - (clock_time - 45:00)`
   - The 40:00-50:00 window is ambiguous and skipped. Median over readings collapses noise.
5. `validate_period_starts` — Sanity checks: both halves detected, FH < SH, 45-75 min gap, FH < 20 min into broadcast, SH > 40 min before end.
6. `find_game_periods` — When both kickoff frames are known, uses `second_half_start - 1` as `first_half_end` (topology-independent, handles broadcasts with few segments).

**Clock ROI:** `CLOCK_SEARCH_ROI = (30, 30, 650, 140)` — wide enough to cover all three 1.CFL scoreboard layouts (clock left / clock right / lower-vertical).

**Verified unsupervised on all 16 matches.** Result: 15 PASS + 1 WARN + 0 FAIL. The one WARN is **ars-dec**, which genuinely has a recording gap: the second-half recording starts when the match clock already shows 48:14, so the first 3:14 of SH gameplay is missing from the source video.

**API:**
```python
from src.broadcast import (
    analyze_broadcast,        # full cut+segment+scoreboard analysis
    detect_period_starts,     # uniform-sample clock OCR, returns kickoff frames + validation
    validate_period_starts,   # standalone sanity checker
    find_game_periods,        # combines kickoffs into GamePeriod objects
)
```

`detect_period_starts` needs only `segments[-1].end_frame` for total_frames — you can skip `analyze_broadcast` entirely and pass `[Segment(0, total_frames)]` for a fast OCR-only path (~5s/match vs ~7min/match for the full cut-detection pipeline).

## Pitch Homography — PnLCalib (pretrained model)

Maps image pixels to pitch coordinates (meters on a 105x68m FIFA-standard pitch). Uses **PnLCalib**, a pretrained HRNet-based encoder-decoder that detects keypoints and lines on the pitch, then estimates full camera parameters (3x4 projection matrix).

**Why PnLCalib over classical approach:** The classical pipeline in `src/homography.py` (Hough lines + line classification) was tested on 5 matches × 3 frames = 15 frames and produced 0 visually correct results — even when reprojection error was low, projected player positions were wrong. Fundamental limitations: Hough transform fragments curves (center circle, penalty arcs) into noise, and line identification heuristics fail without clear perpendicular line structure. A pretrained model was chosen instead.

**Model comparison (tested on 5 diverse frames):**
| Model | Success rate | Notes |
|---|---|---|
| **PnLCalib** (chosen) | 3/5 excellent | Fails cleanly (returns None). Best quality when it works. Outputs 3D-aware 3x4 projection matrix. |
| YOLO Keypoint (Soccana) | 2-3/5 decent | More aggressive — attempts every frame but produces wrong results rather than admitting failure. |
| SoccerNet sn-calibration | not tested | PnLCalib is its SOTA successor on same benchmarks. |

**Current implementation: `run_pnlcalib_video.py` (v2)**

Standalone script that runs PnLCalib on a match clip with five post-processing layers:

1. **Non-gameplay filtering** — Checks for scoreboard overlay via edge density in the `SCOREBOARD_ROI`. Frames without scoreboard (replays, close-ups, graphics) are skipped entirely — no homography is attempted, preventing false positives on non-pitch imagery. Threshold is auto-calibrated per match from the first 10 sampled frames.

2. **Player sanity check** — Projects all detected player feet (via YOLO) to pitch coordinates. Rejects projections where <50% of players land within the 105×68m pitch (5m margin). Catches completely wrong calibrations.

3. **Line alignment check** — Projects a set of key pitch lines (touchlines, halfway line, PA edges) and checks how many sampled points land on bright pixels in the image. Uses HSV thresholding (S<80, V>140) with large dilation. Catches projections that pass the player check by coincidence but are visually wrong (e.g., lines on stands or buildings).

4. **Temporal consistency with outlier rejection** — The `ProjectionSmoother` has three response modes:
   - **Small change** (diff < 0.08): Normal camera movement. EMA blend (alpha=0.3) for smooth transitions.
   - **Medium change** (0.08 < diff < 0.3): Suspicious. Requires 3 consecutive frames agreeing on the new direction before accepting. Single-frame glitches are rejected and the last stable projection is reused.
   - **Large change** (diff > 0.3): Likely a camera cut. Also requires 3-frame confirmation before resetting.
   - When no valid P is available, reuses last good projection for up to 3 seconds (75 frames at 25fps).

5. **Sampled line drawing** — Long pitch lines (touchlines, sidelines) are sampled at intermediate world-space points before projection, so the visible segment is drawn even when endpoints are far off-screen.

**Results — tested on all 16 matches (v2, 60-second clips at 10:00 into first half):**

| Tier | Matches | Coverage | Notes |
|---|---|---|---|
| **Tier 1 (>80%)** | jez-ars, dec-mla, jez-jed, sut-pet, sut-mla, mla-bud-2, jed-ars | 84-100% | Lines visually correct, stable across frames |
| **Tier 2 (60-80%)** | pet-mor, pet-bok, bok-jed, bud-sut | 64-75% | Correct when present; gaps from close-ups/replays |
| **Tier 3 (<60%)** | mor-bud, mor-ars, ars-dec | 10-43% | Night/dusk games, oblique cameras, running tracks |
| **Crashes** | bok-jed-2, mla-bud | — | OpenCV assertion error (video codec/resolution issue) |

Average coverage: **71.3%** (up from 64.7% in v1). All matches improved or stayed flat vs v1.

**Failure mode analysis:**
- **Night/dusk** (mor-ars, mor-bud): PnLCalib can't see pitch markings in low light. Model limitation.
- **Oblique camera** (ars-dec): Camera angle outside PnLCalib's training distribution (top European leagues). 1.CFL's small stadiums have lower, more oblique cameras.
- **Running tracks** (mor-ars): Model sometimes locks onto track lane markings instead of pitch lines.
- **Close-ups/replays** (bud-sut, mla-bud-2): Now correctly filtered by scoreboard check. Coverage % reflects gameplay frames only.

**Optical-flow camera motion fallback (`src/camera_motion.py`).** When PnLCalib produces a trusted projection, `CameraMotionTracker.seed()` detects good features on the pitch (`cv2.goodFeaturesToTrack`) and assigns each a pitch-world coordinate via the inverse plane homography. On subsequent frames where PnLCalib fails, `propagate()` runs Lucas-Kanade flow, RANSACs a fresh `image -> pitch` homography from the surviving correspondences, and returns a 3x4 P packed from that 3x3 (zero z-column — z=0 lines render correctly, goal posts skipped). Features are re-detected every 60 frames to prevent drift, and the trail is reset on non-gameplay frames or confirmed camera cuts. Only the planar homography is tracked: tracked features live on z=0 by construction, so PnP would be coplanar-degenerate, and the downstream event pipeline only needs player-foot -> pitch-XY.

**Manual ground-truth widgets (`src/manual_calibration.py`).** For Tier-3 matches where PnLCalib rarely succeeds — and for ground-truth labelling of any frame in `04_demo_video.ipynb` — three ipywidgets UIs share the same on-disk schema (`data/manual_calibration/{slug}_frame_{N:07d}.json`). All three feed `save_*_calibration`, which fits `H` via the appropriate solver, packs a PnLCalib-convention 3x4 via `projection_from_natural_homography`, and persists. `load_match_seeds(slug)` reads every saved P for a match; the runner uses the frame-index-nearest seed as a last-resort input to the optical-flow tracker when PnLCalib and flow both fail.

  - **`build_labeling_widget`** — original click-landmark UI (`02_homography.ipynb`). User taps named landmarks (corners, halfway/center-circle intersections, PA + 6yd corners, penalty spots — 26 total) on the broadcast frame; `compute_homography` fits via `cv2.findHomography` (RANSAC ≥6 pts, exact LS below). Best when several landmarks are visible.
  - **`build_line_labeling_widget`** — drag-line + tap-point UI (`04_demo_video.ipynb`). User drags two clicks along each painted pitch line to define it (line equation only — clicks don't have to be precise endpoints), and taps single arc-intersection points where straight lines are scarce (e.g. center-circle ∩ halfway, penalty-arc ∩ PA-front). `homography_from_lines_and_points` solves a unified SVD: each line and each point contributes 2 constraint rows on `M = H^(-T)`. Needs ≥4 points-only or ≥5 mixed correspondences. Useful for frames with no clear keypoint landmarks.
  - **`build_line_adjust_widget`** — line-adjust UI seeded by the pipeline's current `P` (`04_demo_video.ipynb`). Pre-projects every named pitch line onto the frame using the supplied initial P; the user clicks a line to confirm it (snaps to the pipeline's projection) and drags an endpoint handle to nudge it onto the painted line. "Confirm all" mass-confirms; live cyan-dashed preview shows the fit. Saves via the same line-correspondence solver. Fastest path when the pipeline is mostly right and only a few lines need correction.

**Dependencies:**
- PnLCalib at `models/pnlcalib/` (gitignored, ~640MB)
- Weights: `models/pnlcalib/weights/SV_kp` (~253MB) and `models/pnlcalib/weights/SV_lines` (~253MB)
- Backup: Soccana YOLO keypoint model at `models/pitch_keypoints/soccana_keypoints.pt` (85MB)

**Usage:**
```bash
# Default: PnLCalib + optical-flow fallback + auto-loaded manual seeds
python -m src.run_pnlcalib_video --match dec-mla --offset_min 10 --duration_sec 60 --alpha 0.3

# Benchmark against the v2 baseline
python -m src.run_pnlcalib_video --match dec-mla --no_flow --no_manual_seeds
```

**Runner fallback chain per frame:** PnLCalib -> sanity checks -> (optical-flow propagate if trail alive) -> (nearest manual seed if any) -> ProjectionSmoother stale hold (up to 3s). Stats breakdown printed at the end splits frames into PnLCalib-success / flow-carry / manual-seed / stale-hold / no-projection.

**Classical pipeline (`src/homography.py`) is retained** but no longer the primary approach. It may be useful as a fallback or for debugging, but should not be relied on for production results.

## Demo Video Pipeline (`src/run_demo.py`, `notebooks/04_demo_video.ipynb`)

End-to-end clip renderer (legacy showcase artefact — originally built for a May 2026 presentation that is no longer a driver; kept because it shares perception code with the pipeline and is useful for eyeballing): tracked players with jersey-coloured ellipses, name + jersey-number badges, live km/h + cumulative distance, ball triangle, pitch-line homography overlay, and a bottom-right minimap.

**Per-frame pipeline:**
1. YOLO detection + BoT-SORT tracking (`persist=True`).
2. Wide-shot gate (`is_wide_shot`): drops close-ups so the flow tracker doesn't reseed on a pointless view; speed display freezes on close-ups but cumulative distance keeps showing.
3. Homography via the same fallback chain as `run_pnlcalib_video` (PnLCalib → flow propagation → manual seed → smoother stale hold), with `refine_projection_to_lines` snapping P onto painted white pixels and a `ProjectionSmoother` (alpha=0.3) EMA-blending across frames.
4. `project_foot` (bottom-centre of bbox) → pitch XY in metres for each player; fed to a `PlayerStats` accumulator with telemetry guards (40 km/h hard cap, 3 m teleport snap, 0.12 m noise floor) so a bad single-frame projection or ID swap doesn't poison cumulative distance. Pre-roll runs through the loop silently so stats are warm by the first rendered frame.
5. Ball: 2-frame velocity-based extrapolation across detection gaps (up to 20 frames ≈ 0.8s @ 25fps), with damping that fades the predicted bbox as the gap grows.

**Per-game configuration in `04_demo_video.ipynb`:**
- `PLAYER_NAMES: dict[track_id, (name, team_id, jersey_no)]` — populated by running the `--discover` CLI first, eyeballing track IDs, then mapping each to a real player. Multiple track IDs can map to the same player (re-entries get fresh BoT-SORT IDs).
- `TRACK_REMAPS: list[(seconds_into_render, {old_id: new_id, ...})]` — patches mid-clip ID swaps when two crossing players' BoT-SORT IDs flip.

**Iteration loop (cells 12-18):** Label rendered frames as Good / Drift / Wrong scale / Wrong overlay / No overlay / Skip via a navigable widget; auto-exports bad frames to `output/demo/diag/labeled/`. Pick a labelled frame and calibrate ground truth on the corresponding source frame using `build_line_adjust_widget` (or the from-scratch `build_line_labeling_widget` if PnLCalib fails on that frame). The compare cell loads every saved GT for the match, runs PnLCalib + refine on that exact source frame, and overlays current pipeline (white) vs ground truth (cyan) with per-frame mean pixel error at 7 reference world points (centre spot, halfway-line endpoints, PA outer corners). Saved GT files double as runtime seeds for the next render via `load_match_seeds`.

**Output layout:**
- `output/demo/{slug}_{ts}_demo.mp4` — final rendered clip
- `output/demo/{slug}_{ts}_discover.mp4` — `--discover` mode: track IDs only, no homography, no names (used to populate `PLAYER_NAMES`)

**CLI usage:**
```bash
# Step 1 — find track IDs
python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16 --discover

# Step 2 — edit PLAYER_NAMES in run_demo.py, then render
python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16
```
The notebook (`04_demo_video.ipynb`) is the iteration-friendly version of the same pipeline — same logic inline in cell 9, plus the label / calibrate / compare cells.

## Game State & Event Detection (`src/game_state.py`, `src/pipeline.py`, `src/events.py`)

The **keystone** that unblocked event detection. Previously every notebook re-ran the
whole expensive perception pass (YOLO + ResNet + PnLCalib) from scratch, so there was
never a stable artifact to build event logic on. Now perception runs **once** and is
persisted; analysis is instant and needs no video. This is the SoccerNet
Game-State-Reconstruction intermediate representation (per-frame players with pitch XY +
team, ball, camera), applied to our custom 1.CFL-tuned stack.

**`game_state.py` — the artifact.** Per-HALF Parquet under `output/game_state/{slug}/p{period}/`
(legacy flat `{slug}/` dirs still load; `available_periods(slug)` lists what exists):
- `players.parquet` — one row per (frame, track_id): `team_id`, image bbox, `foot_xy_img`,
  `pitch_x/pitch_y` (metres on 105×68, NaN if no projection), `conf`.
- `frames.parquet` — one row per frame: `time_sec` (relative to THAT half's kickoff —
  period-2 timestamps restart at 00:00), `period`, `is_gameplay`, `is_wide_shot`,
  `has_P`, `homog_source`, `homog_conf`, flattened 3×4 `P00..P23`, and the *best raw* ball
  detection (`ball_pitch_x/y`, `ball_source` ∈ detected/none).
- `ball.parquet` — one row per raw ball candidate detection (`frame, x1..y2, conf`); a
  frame can have 0..n. This is what `ball_tracker.py` consumes.
- `embeddings.npz` — per-track mean ResNet embeddings (for the future ReID upgrade).
- `meta.json` — slug, video, fps, period, frame range, `partial` flag,
  `homog_source_counts`, track/team counts.
- `GameState.load(slug, period=N)` returns the DataFrames + helpers (`get_P`, `players_at`,
  `ball_trajectory`, `ball_candidates`) + `gs.dir`/`gs.period`; `period=None` auto-resolves
  only when unambiguous (single stored period or legacy flat artifact).

**`pipeline.py` — `PerceptionPipeline`.** Extracts the per-frame loop that was **duplicated**
in `run_demo.py` / `run_pnlcalib_video.py` and writes a `GameState` instead of rendering.
It **imports** the proven pieces (does not reinvent them): `yolo.track(persist=True)`;
the homography fallback chain (`predict_one_frame` → `CameraMotionTracker.propagate` →
`refine_projection_to_lines` → `ProjectionSmoother`, gated by `check_projection_sanity` /
`check_line_alignment` / `is_gameplay_frame`), storing `_line_alignment_score` as
`homog_conf`; `project_foot` / `is_wide_shot`; and the saved `TeamClassifier`
(sampled ResNet embeddings → track-level majority vote, joined back after the pass).
Writes a **complete partial artifact every 15k frames** (`--checkpoint_every`, 0=off) so a
crashed multi-hour run keeps everything up to the last checkpoint (a 4h run died at 90%
once). Track ids restart per run — p1 and p2 track id spaces COLLIDE; anything match-level
must namespace them (the SB export offsets period-2 ids by `PERIOD_TID_OFFSET`).
Run: `python -m src.pipeline --match sut-mla --half 2 --offset_min 0 --duration_sec 2981 --pnl_stride 3`.
Long runs: launch DETACHED from the harness (a harness-tied background run gets orphaned or
killed when the session ends), then `python -m src.stabilize --match X --half N --apply`.

**`events.py` — rule-based events.** Possession-then-event decision tree (Anzer/Bauer,
PLOS One 2024): ball carrier per frame (nearest player within `POSSESSION_RADIUS_M`,
ball slower than `CARRIER_MAX_BALL_SPEED`, not stale-coasted, not dead-ball) **plus
explicit kick detection** (`detect_kicks`: ball-velocity discontinuity ≥ `KICK_DV_MS`
next to a player — catches running one-touches proximity logic can't see) → debounced
*touches* (each must overlap ≥1 *detected*-source ball frame — pure-interpolation touches
are fabrications) → **possession spells** (team gains possession only if spell ≥1s or ≥2
touches; deflections don't flip possession) → Pass / Carry / Shot + Ball Recovery at real
spell boundaries. Dead-ball logic (`dead_ball_frames`) suppresses carrier/kicks from when
the ball crosses the boundary until the restart. Export is **StatsBomb-v4-shaped**
(type ids, possession sequences, Ball Receipt* after completes, pass length/angle,
outcomes omitted on completes), attack-normalized, with player stats keyed to
consolidated meta-tracks (`player-m{N}`; real names when `data/identities/{slug}.json`
exists; `--raw_tracks` disables consolidation).

**Match-level export.** `python -m src.events --match X` with no `--half` detects events
on every stored period and merges them into ONE `output/events/{slug}_events.json`:
running event index, possession numbering continues across halftime, period-2 minutes
+= 45 (timestamps restart per period), period-2 player ids offset by
`PERIOD_TID_OFFSET=100000` and suffixed `-h2` (no cross-half ReID yet), a halftime
attack-direction-flip sanity check, and per-period summaries under
`summary.periods_detail`. Oracle goals (below) are injected as Shot/outcome Goal (id 97).
`--half N` exports a single period to `{slug}_p{N}_events.json` instead.

## Scoreboard Goal-Oracle (`src/score_ocr.py`) & Match Report (`src/report.py`)

Goals cannot be trusted from tracking data alone — QC showed the sut-mla goal happened on
a close-up camera where events are correctly paused. The broadcast graphics announce the
score instead. Two scoreboard families across the league (all six surveyed matches parse):
score as one text token in a dark box ("JED 0-0 ARS 28:10" — jed-ars, jez-jed, dec-mla,
mor-bud) read by generic OCR, and score as two digits in auto-located red boxes
("25:24 PET [0][1] MOR" — sut-mla, pet-mor) needing a digit-allowlist read of the
4x-upscaled box crop. Visibility is intermittent everywhere, so the scoreboard is sampled
on a fixed 10s cadence (a sample either parses or is discarded); the bottom red banner
(kickoff/goal/HT/FT) is swept by red-fraction and OCR'd per run. Misreads are
majority-filtered; each monotonic score step is a **certain goal**. Timing anchor
(pixel-calibrated): tight bracket (≤30s) → the last old-score reading (~1s from the real
goal; the operator updates AFTER the ball crosses); wide bracket → first sighting of the
new score. Readings are cached (`{slug}_score_readings.json`), so re-derivation is
instant. Team mapping is manual: `--home_team {0|1}` (home = left team on the graphics;
verify via kit colors — sut-mla home team1 = Sutjeska blue). Validated on sut-mla:
final score OCR 1-0 == FT banner == real result; bracket [97650, 97900] contains the
verified ball-in-net frame at its left edge.

`report.py` renders the one-page analyst deliverable from the merged JSON (mplsoccer):
score header (club names from slug via `CLUB_NAMES`), stats block, pass-volume momentum
strip with goal stars + HT line, per-team pass maps (attack-normalized →), shot/goal map
(goal locations marked as estimated), top passers (consolidated ids or real names) →
`output/reports/{slug}/{slug}_match_report.png`.

Run order per match: perception both halves → stabilize both → `score_ocr --home_team N`
→ `events` (merged) → `report`. First complete match: **sut-mla, 1,434 events, 877
passes, possession 54/46, goals 1-0 (matches the real result)**.

**`golden_eval.py` + `data/golden_events/` — ground truth.** Hand-labeled ball-control
intervals + passes (built by frame-stepping contact sheets of ball-centered crops).
Scoring vs golden on sut-mla (2×40s): **pass precision 0.75, recall 0.82, carrier
accuracy-when-assigned 99%**; residual errors are one situation class (throw-in/restart
handling — the out-of-bounds crossing often isn't registered because homography error
near the touchline keeps the ball estimate inside) plus the structural **GK gap** (GKs
classified 'Other' are invisible to possession — the golden GK build-up pass is a known
FN). Unknown spans (aerial balls the labeler couldn't see) score separately as
'unverifiable'. Tune against this, not against aggregate plausibility. Exports `output/events/{slug}_events.json` — a `summary`
block (possession %, passes, completion %, shots, turnovers) + per-event StatsBomb-lite
records (`location` in 120×80, native `pitch_xy` in metres, type-specific detail block).
Run: `python -m src.events --match sut-mla`. QC in `04`→ `03_event_detection.ipynb`
(pass-map + possession bar).

**Generalization fixes (July 2026, found by running the chain on `dec-mla` + `bud-sut`):**
- **ProjectionSmoother confirmation deadlock.** On a continuously fast-panning camera
  (dec-mla's small ground) every consecutive raw P differs by more than the outlier
  threshold, so the 3-frame confirmation never fires and the smoother returns stale state
  forever — the artifact stored garbage labeled `pnlcalib` for minutes while *fresh*
  PnLCalib on the same frames was visually perfect. Fix: `ProjectionSmoother.reset_to` +
  a fresh-beats-stale snap in `pipeline._homography` (if the gate-verified fresh P scores
  ≥1.5x the smoothed output's line-alignment on the same pixels, adopt it).
- **Homography trust threshold is match-dependent.** The line-alignment score separates
  good from bad *within* a match but its scale varies with paint/lighting (floodlit paint
  has a green cast: S~115 vs sut-mla's ~35, so perfect dec-mla projections score 0.12-0.53
  and a fixed 0.75 gate rejected 100% of the match). Do NOT "fix" this by loosening the
  white-pixel mask — an adaptive-V mask was tried and inflated a known-WRONG sut-mla frame
  from 0.64 to 1.00. Fix: `game_state.adaptive_conf_min` — per-match threshold =
  0.75 x q75(conf of fresh-PnLCalib wide frames), clamped to [0.15, 0.75]. sut-mla
  resolves to exactly 0.75 (bit-identical events); `trusted_frame_mask` lives in
  `game_state.py` and is shared by events + ball tracker + renderer.
- Perception speed: `predict_one_frame` accepts the pipeline's already-computed
  `player_boxes` (was running YOLO twice per wide frame); `--pnl_stride N` (opt-in) runs
  PnLCalib every Nth wide frame with gated optical flow carrying between.

**Validated on `sut-mla`** (60s @ 10:00 FH, then re-validated on a 4-min window with the
Kalman tracker) with a full **visual QC pass** (renders in `output/qc/sut-mla/`):
- **Homography — coverage was inflated.** Raw `has_P` = 96.6%, but eyeballing the overlays
  showed two failure classes counted as "covered": (a) close-up shots get a wide-shot P
  carried onto them by the stale/manual-seed fallbacks (garbage), and (b) a mid-confidence
  band (~0.5–0.75) with geometrically-wrong projections that still pass the automated
  line-alignment gate (e.g. f18586 @ conf 0.64: penalty arc ~500 px off the paint). Honest
  *correct* coverage ≈ **70–77%**. Genuine wide frames (conf ≥ ~0.85) are accurate (centre
  circle / penalty box verified). Fix applied: `events.py` gates on
  `is_wide_shot AND homog_conf ≥ HOMOG_CONF_MIN (0.75)`; `pipeline.py` no longer attempts
  homography on non-wide frames.
- **Team classification — verified correct** (~94%): objective kit-hue cross-check gives
  team0=Mladost(yellow) 150/3, team1=Sutjeska(blue) 113/13. Consistent, not the skew driver.
- **Possession 98/2 on the slice is REAL** (Mladost dominated that minute), *not* a
  ball-tracking artifact — confirmed by overlaying ball+carrier (ball genuinely at the
  yellow player's feet). A 60 s clip is not a match aggregate.

**`ball_tracker.py` — Kalman ball tracker (Milestone 2, done).** QC found YOLO loses the
ball on ~45% of frames (gaps of 12–64 frames, exactly during passes); the old 2-frame
image-space extrapolation left damped ghost positions hovering at the kicker, which merged
passes into one long touch and even fabricated possession for the wrong team. Replacement:
a **pitch-space** constant-velocity KF over the persisted artifact (pure numpy/pandas — no
video/GPU, so parameters iterate in seconds). Pitch space matters: the camera pans to follow
the ball, so image-space coasting is wrong during exactly the gaps that need bridging, while
the *ground track* of an airborne pass is genuinely constant-velocity and camera-independent.
Key mechanics: measurements = candidates from `ball.parquet` projected via trusted P only
(same `is_wide_shot AND homog_conf ≥ 0.75` gate as events); Mahalanobis gating with
motion-consistent candidate selection; birth/confirm hysteresis (lone false positives never
reported); kick handling via consistent-rejection reinit; honest coasting with a staleness
gate (`events.py` refuses possession from a ball unseen > `CARRIER_MAX_MISSED` frames); and
a **hindsight bridging pass** — coast/dead runs bounded by detections ≤ 2.4s apart are
rewritten as a straight line (`source='bridged'`), which visual QC confirmed lands within
~1–2 m of the real ball mid-blackout. On the sut-mla 4-min window: ball coverage within
trusted frames ~90%, speeds physical (p95 = 15.5 m/s), team-1 passes recovered 4→7,
ghost-derived possession eliminated. Run: `python -m src.ball_tracker --match sut-mla`.

## Dataset Conventions

- **Filename format:** `{match-slug}_frame_{frame_number:07d}.jpg` (e.g., `bud-sut_frame_0012345.jpg`)
- **Match slugs:** Short identifiers like `bud-sut`, `ars-dec`, `mla-bud-2` (see `Config.MATCH_VIDEOS` for all 16)
- **Label format:** YOLO normalized `class cx cy w h` per line
- **Train/val split:** Match-level (entire matches held out), NOT random frames. Val matches: `mla-bud-2`, `jed-ars`, `pet-mor`. Prevents data leakage.
- **Master copy:** `data/object_detection/images/all/` and `data/object_detection/labels/all/` hold the full 971-frame dataset. `train/` and `val/` are populated by `split_val()`.
- **Annotations:** Done in CVAT, exported as YOLO 1.1 format

## Key Configuration (src/config.py)

- `YOLO_MODEL = None` - auto-resolves to latest trained weights
- `YOLO_SEG_MODEL = "yolov8n-seg.pt"` - segmentation for team classification
- `PLAYER_CONF_THRESHOLD = 0.25`
- `BALL_CONF_THRESHOLD = 0.15`
- `TRACKER_TYPE = "botsort"`
- `TEAM_COLORS = {0: (255, 50, 50), 1: (0, 220, 255)}` - BGR
- `MATCH_VIDEOS` dict maps all 16 match slugs to video file paths
- `CLOCK_SEARCH_ROI = (30, 30, 650, 140)` - covers all three 1.CFL scoreboard layouts

## Known Limitations

- **Team classification** works well on ~10/16 matches, struggles on ~6 where jersey colors are similar or lighting is difficult. All 16 matches labeled and validated via 2-min annotated clips. Majority vote per track is more robust than a single median-embedding prediction but ID swaps from long occlusions can still cause systematic errors.
- **Referee/GK filtering** — GKs are now identified positionally (`src/roles.py`: deepest-player-in-frame ≥70% + goal-zone residence) and folded into possession with the defending team; refs are excluded by the same signature. The old GMM-probability idea is superseded.
- **Shots are conservative; goals come from the oracle** — direction-aware validation removed all 7 nearest-goal artifacts; the detector now finds only clear on-target attempts (1 on the sut-mla full match, pixel-verified real). Goals are supplied by the scoreboard goal-oracle with certainty but estimated location and unknown scorer; shot outcomes other than Goal remain Unknown.
- **No cross-half outfield identity** — track ids restart per half; the merged export namespaces period-2 ids (`player-mN-h2`), so one outfield player is two rows across halves. GKs ARE unified (`goalkeeper-t{N}`, via roles). The persisted team-classifier embeddings were measured to carry NO within-team identity signal (P(same-player pair closer)=0.43) — do not build ReID on them; naming both halves via the identity widget unifies players by name today; jersey-number OCR is the planned automatic path.
- **Restart over-detection** — the sticky dead-ball trigger finds ~3x the true throw-in count (homography drift parks the ball estimate on the line). Golden shows this does not hurt pass precision, but play_pattern tags are over-applied. Tightening needs a better out-of-bounds signal than position alone.
- **Tier-2 conditions degrade the chain measurably** (bok-jed golden baseline: pass P/R 0/0, carrier 26%/51%): homography-untrusted stretches blank out whole control intervals, crowds misattribute the carrier via the lagging ball estimate, and roles need at least a half of data. Expect sparse events on Tier-2 batch output.
- **Goal-oracle team mapping is manual** — pass `--home_team {0|1}` per match (graphics say home/away; classifier team indices are arbitrary).
- **Throw-in/restart semantics half-handled** — dead-ball suppression works when the ball's tracked position crosses the boundary, but homography error near the touchline sometimes keeps the out-ball "inside", minting restart-period false passes (golden segA class).
- **Pitch homography** — PnLCalib v2 averages 71% coverage across 16 matches. Tier 1 matches (7 of 14 working) are at 84-100%. Tier 3 matches (3 of 14) are at 10-43% due to night/dusk conditions, oblique cameras, and running tracks. Two matches crash with OpenCV errors. No camera motion estimation yet — each frame is processed independently.
- **Ball tracking** (`ball_tracker.py` KF) can only bridge gaps bounded by trusted detections; a kick followed by a blackout *and* no re-acquisition within 2.4s still loses the ball, and airborne balls project with overshoot while high (z=0 plane assumption). Possession during long blackouts is honestly unknown — a real opponent touch inside one is missed, not misattributed.
- **ars-dec source video** has a ~3:14 recording gap at the start of the second half — the second-half recording begins with the match clock already at 48:14. Period detection correctly flags this via its 42-min gap warning, but downstream event detection on ars-dec SH will be missing the first ~3 min of gameplay.

## Conventions

- All image processing uses **BGR** (OpenCV default). Convert to RGB only for display (matplotlib, PIL).
- `src/` modules use relative imports (`from .config import Config`). Notebooks add `sys.path.insert(0, str(Path.cwd()))` and use `importlib.reload()` for hot-reloading during development.
- The 16 matches are all from Montenegro's 1.CFL (First Football League), seasons 25/26.
