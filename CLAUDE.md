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
- **PCA + GMM** (scikit-learn) - dimensionality reduction (512->32) then 2-component Gaussian Mixture Model for team clustering
- **easyocr** - match clock reading for period detection
- **OpenCV** - video I/O, image processing, homography estimation

## Project Structure

```
football-computer-vision/
├── main.ipynb                 # Main pipeline: detection, tracking, team classification, export
├── model_fine_tuning.ipynb    # YOLO training + error analysis
├── src/
│   ├── __init__.py            # Lazy imports for all modules
│   ├── config.py              # All paths, thresholds, hyperparameters
│   ├── detection.py           # PlayerDetector (YOLO), BallInterpolator
│   ├── segmentation.py        # YOLOv8-seg masks, ResNet18 CNN embeddings
│   ├── team_classifier.py     # TeamClassifier: PCA + GMM clustering
│   ├── tracking.py            # Tracker: track-level team assignment
│   ├── broadcast.py           # Camera cut detection, period detection via clock OCR
│   ├── homography.py          # Pitch line detection, homography estimation
│   ├── visualization.py       # Annotator: ellipses, triangles, bboxes
│   ├── video_utils.py         # Video I/O, clip extraction (ffmpeg)
│   └── dataset.py             # Dataset management, train/val split
├── dataset/
│   ├── data.yaml              # YOLO dataset config (2 classes)
│   ├── images/{all,train,val}/  # 971 annotated frames
│   ├── labels/{all,train,val}/  # YOLO-format labels (class cx cy w h)
│   └── cvat_exports/          # Archived CVAT annotation exports
├── runs/detect/               # YOLO training outputs (weights, curves)
├── videos/                    # Source match videos (gitignored)
├── output/                    # Annotated output videos (gitignored)
├── period_detection_fast_results.json  # Cached period boundaries for all 16 matches
├── requirements.txt
└── README.md
```

## Roadmap (in priority order)

1. ~~**Broadcast analysis**~~ — Done. Unsupervised period detection works on all 16 matches (15 PASS, 1 WARN).
2. **Pitch homography** — In progress (`src/homography.py`). Works on ~40% of frames with <1.5m error. Needs center circle detection and temporal smoothing.
3. **Event detection** — Ball position + player positions + team labels + pitch coords -> detect passes, shots, possession changes. Core deliverable. Blocked on homography.
4. **Cross-shot player matching** — Re-identify players across camera cuts using team + pitch position (~70-80% automated, human corrects the rest).
5. **Jersey number / player identity** — Semi-manual: human maps track clusters to squad lists. Full OCR is unreliable at broadcast resolution.

## Data Flow (main.ipynb batch pipeline)

The batch annotation cell processes each game in two passes:

```
Pass 1: Detect + Track + Sample Embeddings
  For every frame:
    YOLO detection + BoT-SORT tracking -> all_detections[]
  Every 15th frame:
    YOLOv8-seg (full frame) -> person masks
    Masked crops -> ResNet18 -> 512-dim embeddings
    Store per-track: track_embeddings[track_id].append(embedding)

Between passes: Fit & Classify
  All embeddings -> PCA (512->32) -> GMM (k=2) fit
  Per track: mean(embeddings) -> GMM predict -> team_id
  BallInterpolator fills gaps up to 15 frames

Pass 2: Annotate
  Re-read video, draw team-colored ellipses + ball triangles -> MP4
```

**Key design choice:** Team classification is done at the **track level**, not per-frame. Each track's embeddings are averaged into one vector and classified once. This eliminates flickering.

## Detection Model

- **Architecture:** YOLOv8m (25M params)
- **Classes:** 2 - `person` (0), `ball` (1)
- **Training:** 150 epochs, imgsz=1280, batch=8, mosaic=1.0, copy_paste=0.3, patience=30
- **Dataset:** 971 frames across 16 matches (822 train / 149 val, match-level split)
- **Best model:** `runs/detect/football_2class_yolov8m_v22/weights/best.pt`
- **Metrics:** mAP50=0.944, mAP50-95=0.748, Precision=0.952, Recall=0.892
- **Weight resolution:** `Config.resolve_yolo_model()` auto-finds the latest `runs/detect/*/weights/best.pt`

Roles like goalkeeper, referee, and team assignment are NOT part of detection — they're handled entirely in post-processing (team classifier + future position heuristics).

## Team Classification Pipeline

1. **Segmentation:** YOLOv8n-seg runs on the full frame (not crops!) to produce person silhouette masks
2. **Masking:** Each player's crop is masked - background pixels replaced with mean foreground color to avoid CNN bias
3. **Embedding:** Frozen ResNet18 backbone produces 512-dim vectors per masked crop, resized to 128x64 (tall format for full-body players)
4. **PCA:** 512 -> 32 dims, removes noise
5. **GMM fit:** Two-pass process:
   - Pass 1: Rough k=2 GMM on all embeddings
   - Pass 2: Remove bottom 5% by membership probability, re-fit for tighter centroids
6. **Track-level prediction:** Average all embeddings for a track, classify the mean embedding once

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

## Pitch Homography (`src/homography.py`)

Maps image pixels to pitch coordinates (meters on a 105x68m FIFA-standard pitch). Classical computer vision approach.

**Pipeline:**
1. `detect_field_mask` — HSV green segmentation -> largest contour = pitch area
2. `detect_line_pixels` — Dual-channel white line detection within the field mask:
   - Color-based: bright (adaptive threshold based on field median) + low saturation (<80)
   - CLAHE-enhanced white top-hat transform: highlights thin bright structures on darker background
   - Player bounding boxes (from YOLO) masked out before detection to eliminate white jersey false positives
   - Adaptive thresholds: lower for dark fields (night/dusk games, detected via median V < 100)
   - Connected component filtering: keeps elongated shapes, rejects compact blobs
3. `detect_lines` — Probabilistic Hough Transform (threshold=40, minLineLength=60, maxLineGap=20)
4. `merge_lines` — Co-linear segments merged (angle < 10deg, perpendicular distance < 50px, validated against white pixel support). Post-merge filter removes lines < 100px.
5. `classify_lines` — Semantic labeling based on angle:
   - **touchline** (< 15deg from horizontal): runs along the pitch length
   - **perpendicular** (> 30deg from horizontal): halfway line, PA edges, goal lines
   - **other** (15-30deg): ambiguous, could be PA top/bottom under heavy perspective
6. `identify_touchlines` — Near (highest y = bottom of frame) and far (lowest y) touchlines, with minimum length (15% of frame width) and separation (10% of frame height) checks.
7. `_identify_perpendicular_lines` — Matches perpendicular lines to known FIFA pitch lines using:
   - Spacing patterns between detected lines vs known pitch geometry (PA edge = 16.5m, halfway = 52.5m, etc.)
   - Position heuristic for single-perp fallback (center of frame = halfway line)
   - Scoring: prefers assignments where halfway line is near frame center and pixels-per-meter is reasonable (~10-20)
8. `estimate_homography` — Builds image<->pitch correspondences from touchline x perpendicular intersections, computes homography via RANSAC (threshold 5.0m), validates with player foot projection sanity check.

**Current results** (tested on 5 matches x 3 frames = 15 frames):
- 6/15 produce a working homography (~40%)
- All successful homographies have < 1.5m reprojection error on inlier points
- Best results on daylight games with clear pitch markings

**Failure modes (in priority order for fixing):**
1. **Insufficient perpendicular lines** (most common): Midfield views without penalty area structure lack x-coordinate anchors. Both touchlines are detected but there's nothing to establish the horizontal position on the pitch.
2. **Wrong line identification**: The spacing/position heuristic sometimes misidentifies which pitch line a detected perpendicular corresponds to. RANSAC catches some of these but not all.
3. **Unusual stadiums**: jez-jed has a running track between camera and pitch with worn/faint markings — line detection fails.
4. **Night game noise**: Dark fields (mor-ars, dec-mla) amplify grass texture in the line mask, producing many false line detections.

**Highest-impact next improvements:**
1. **Center circle detection** (ellipse fitting) — The center circle is visible in most broadcast frames and would immediately anchor the pitch center (x=52.5, y=34), solving the "no perpendicular lines" problem for midfield views.
2. **Hypothesis testing for line identification** — Try multiple assignments of detected perpendicular lines to known pitch lines, score each by how many players project to valid pitch positions. Would fix wrong-identification failures.
3. **Temporal smoothing** — Propagate a good homography from nearby frames to fill gaps. Camera motion between adjacent frames is small, so the homography changes slowly.

**API:**
```python
from src.homography import estimate_homography, HomographyResult
result = estimate_homography(frame, player_boxes=yolo_boxes)
if result.H is not None:
    # Map pixel coordinates to pitch meters
    mapped = cv2.perspectiveTransform(pixel_pts, result.H)
    # result.reprojection_error = mean error in meters (inliers only)
    # result.n_inliers / result.n_matches = inlier ratio
    # result.lines = list of DetectedLine with labels
```

## Dataset Conventions

- **Filename format:** `{match-slug}_frame_{frame_number:07d}.jpg` (e.g., `bud-sut_frame_0012345.jpg`)
- **Match slugs:** Short identifiers like `bud-sut`, `ars-dec`, `mla-bud-2` (see `Config.MATCH_VIDEOS` for all 16)
- **Label format:** YOLO normalized `class cx cy w h` per line
- **Train/val split:** Match-level (entire matches held out), NOT random frames. Val matches: `mla-bud-2`, `jed-ars`, `pet-mor`. Prevents data leakage.
- **Master copy:** `dataset/images/all/` and `dataset/labels/all/` hold the full 971-frame dataset. `train/` and `val/` are populated by `split_val()`.
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

- **Team classification** works well on ~10/16 matches, struggles on ~6 where jersey colors are similar or lighting is difficult
- **Referee/GK filtering** via bottom-5% GMM membership probability is weak — refs often cluster tightly with one team. Future plan: use pitch position (requires homography)
- **Pitch homography** works on ~40% of frames. The main blocker is perpendicular line detection — center circle detection would unlock most remaining frames.
- **Ball tracking** uses simple linear interpolation across gaps. Could be improved with tracker-aware logic
- **ars-dec source video** has a ~3:14 recording gap at the start of the second half — the second-half recording begins with the match clock already at 48:14. Period detection correctly flags this via its 42-min gap warning, but downstream event detection on ars-dec SH will be missing the first ~3 min of gameplay.

## Conventions

- All image processing uses **BGR** (OpenCV default). Convert to RGB only for display (matplotlib, PIL).
- `src/` modules use relative imports (`from .config import Config`). Notebooks add `sys.path.insert(0, str(Path.cwd()))` and use `importlib.reload()` for hot-reloading during development.
- The 16 matches are all from Montenegro's 1.CFL (First Football League), seasons 25/26.
