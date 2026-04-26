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
│   └── run_demo.py                    # FSCG demo renderer: tracking + names + speed/distance + minimap
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
│   └── demo/                          # FSCG demo MP4s (`{slug}_{ts}_demo.mp4`, `{slug}_{ts}_discover.mp4`)
├── notebooks/
│   ├── 01_team_classification.ipynb   # Per-game labeling widget → KNN classifier + labels.npz + review + validation
│   ├── 02_homography.ipynb            # Homography visualization + model comparison
│   ├── 03_event_detection.ipynb       # Event detection scaffold (passes, shots, possession)
│   ├── 04_demo_video.ipynb            # FSCG demo render + label/calibrate/compare iteration loop
│   └── detection_training.ipynb       # YOLO training + error analysis
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Roadmap (in priority order)

1. ~~**Broadcast analysis**~~ — Done. Unsupervised period detection works on all 16 matches (15 PASS, 1 WARN).
2. **Pitch homography** — In progress. PnLCalib pretrained model + post-processing (v2), Lucas-Kanade optical-flow propagation (`camera_motion.py`), and manual seeds (`manual_calibration.py`) all wired into the runner. v2 averages 71% coverage across 16 matches. The demo notebook (`04_demo_video.ipynb`) added a label/calibrate/compare iteration loop with a line-adjust ground-truth widget seeded from the pipeline's own P. Next step: use that loop on a representative clip to diagnose the residual drift/wrong-overlay failures (PnLCalib output vs `refine_projection_to_lines` bias vs flow-propagated bad seed) and tighten whichever stage is responsible.
3. **Event detection** — Ball position + player positions + team labels + pitch coords -> detect passes, shots, possession changes. Core deliverable. Blocked on homography.
4. **Cross-shot player matching** — Re-identify players across camera cuts using team + pitch position (~70-80% automated, human corrects the rest).
5. **Jersey number / player identity** — Semi-manual: human maps track clusters to squad lists. Full OCR is unreliable at broadcast resolution.

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

End-to-end clip renderer used as a pitch artefact for the FSCG presentation: tracked players with jersey-coloured ellipses, name + jersey-number badges, live km/h + cumulative distance, ball triangle, pitch-line homography overlay, and a bottom-right minimap.

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
- **Referee/GK filtering** via bottom-5% GMM membership probability is weak — refs often cluster tightly with one team. Future plan: use pitch position (requires homography)
- **Pitch homography** — PnLCalib v2 averages 71% coverage across 16 matches. Tier 1 matches (7 of 14 working) are at 84-100%. Tier 3 matches (3 of 14) are at 10-43% due to night/dusk conditions, oblique cameras, and running tracks. Two matches crash with OpenCV errors. No camera motion estimation yet — each frame is processed independently.
- **Ball tracking** uses simple linear interpolation across gaps. Could be improved with tracker-aware logic
- **ars-dec source video** has a ~3:14 recording gap at the start of the second half — the second-half recording begins with the match clock already at 48:14. Period detection correctly flags this via its 42-min gap warning, but downstream event detection on ars-dec SH will be missing the first ~3 min of gameplay.

## Conventions

- All image processing uses **BGR** (OpenCV default). Convert to RGB only for display (matplotlib, PIL).
- `src/` modules use relative imports (`from .config import Config`). Notebooks add `sys.path.insert(0, str(Path.cwd()))` and use `importlib.reload()` for hot-reloading during development.
- The 16 matches are all from Montenegro's 1.CFL (First Football League), seasons 25/26.
