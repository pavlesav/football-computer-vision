# Football Computer Vision

Semi-automated event data extraction from broadcast football footage. The end goal is to produce match event data (passes, shots, possession) for every Montenegro First Division (1.CFL) game, with a human-in-the-loop workflow targeting 1-3 hours of correction per match.

## Pipeline Overview

The system processes a full-match broadcast video through a layered pipeline, where each layer builds on the previous:

```
Full-match broadcast video
    |
    v
[1. Broadcast Analysis]  -->  Game period boundaries (kickoff frames)
    |
    v
[2. Detection + Tracking]  -->  Player/ball bounding boxes + persistent track IDs
    |
    v
[3. Team Classification]  -->  Team A vs Team B labels per track
    |
    v
[4. Pitch Homography]  -->  Pixel coordinates mapped to pitch coordinates (meters)
    |
    v
[5. Event Detection]  -->  Passes, shots, possession changes  (planned)
```

### Current Status

| Layer | Status | Details |
|-------|--------|---------|
| Broadcast Analysis | Done | Unsupervised period detection, 15/16 matches PASS |
| Detection + Tracking | Done | YOLOv8m + BoT-SORT, mAP50=0.944 |
| Team Classification | Done | ResNet18 + PCA + KNN (k=5), 16/16 matches labeled |
| Pitch Homography | In Progress | PnLCalib v2 + optical-flow + manual seeds, 71% avg coverage; per-frame drift being diagnosed via GT loop |
| Event Detection | Planned | Requires homography |
| Demo Video | Done | FSCG clip renderer with tracked players, names, speed/distance, minimap (`04_demo_video.ipynb`) |

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
│   ├── camera_motion.py               # Optical-flow propagation across PnLCalib dropouts
│   ├── manual_calibration.py          # Landmark / drag-line / line-adjust GT widgets + saved seeds
│   ├── homography.py                  # Classical pitch line detection (legacy, not primary)
│   ├── visualization.py               # Annotator: ellipses, triangles, bboxes
│   ├── video_utils.py                 # Video I/O, clip extraction (ffmpeg)
│   ├── dataset.py                     # Training data management, train/val split
│   ├── run_pnlcalib_video.py          # PnLCalib homography on video clips
│   └── run_demo.py                    # FSCG demo renderer: tracking + names + speed/distance + minimap
├── models/                            # All models and weights
│   ├── detection/weights/best.pt      # Fine-tuned YOLOv8m
│   ├── segmentation/yolov8n-seg.pt    # Instance seg for team classification
│   ├── pitch_keypoints/               # Soccana YOLOv11 keypoint model (backup)
│   └── pnlcalib/                      # PnLCalib — primary homography model (gitignored, ~640MB)
├── data/
│   ├── period_detection_results.json  # Cached period boundaries for all 16 matches
│   ├── manual_calibration/            # {slug}_frame_{N:07d}.json — manual homography seeds
│   └── object_detection/              # Labeled training data for YOLO fine-tuning
│       ├── data.yaml                  # YOLO dataset config (2 classes)
│       ├── images/{all,train,val}/    # 971 annotated frames across 16 matches
│       └── labels/{all,train,val}/    # YOLO-format labels (class cx cy w h)
├── notebooks/
│   ├── 01_team_classification.ipynb   # Per-game labeling widget → KNN classifier + review + validation
│   ├── 02_homography.ipynb            # Homography visualization + model comparison
│   ├── 03_event_detection.ipynb       # Event detection scaffold (passes, shots, possession)
│   ├── 04_demo_video.ipynb            # FSCG demo render + label/calibrate/compare iteration loop
│   └── detection_training.ipynb       # YOLO training + error analysis
├── videos/                            # Source match videos (gitignored)
├── output/                            # All generated outputs (gitignored)
│   ├── classifiers/                   # Per-game classifier.pkl + labels.npz + pass1.pkl
│   ├── classifier_validation/         # Annotated 2-min clips for visual QC
│   └── demo/                          # FSCG demo MP4s (full clip + discover-mode track-ID clip)
├── requirements.txt
├── CLAUDE.md                          # Detailed project guide for AI assistants
└── README.md
```

## Layer Details

### 1. Broadcast Analysis (`src/broadcast.py`)

Fully unsupervised segmentation of a broadcast into game periods.

- **Camera cut detection**: HSV histogram difference between consecutive frames with adaptive thresholding. Cut count varies widely by production (4 cuts for single-camera to 586 for multi-camera).
- **Period detection**: OCR of the match clock via easyocr on 60 uniformly-sampled frames. Extrapolates kickoff times from clock readings: `clock < 40:00` maps to first half, `clock >= 50:00` maps to second half.
- **Validation**: Sanity checks (both halves detected, 45-75 min gap, reasonable positions in broadcast). Tested on all 16 matches: 15 PASS, 1 WARN (ars-dec has a source video recording gap).

### 2. Detection + Tracking (`src/detection.py`)

- **YOLOv8m** fine-tuned on 971 frames across 16 matches (2 classes: person, ball)
- **BoT-SORT** tracking provides persistent IDs across frames
- **Ball interpolation** fills detection gaps up to 15 frames via linear interpolation
- Metrics: mAP50=0.944, mAP50-95=0.748, Precision=0.952, Recall=0.892

### 3. Team Classification (`src/team_classifier.py`, `src/segmentation.py`)

One-time labeling per game (~15 min), then applied automatically to all subsequent processing.

1. YOLOv8n-seg produces per-player silhouette masks on the full frame
2. Masked crops (background removed) fed to frozen ResNet18 for 512-dim embeddings
3. PCA reduces 512 → 32 dims
4. Human labels ~200 track crops per game (A/B/Other) in the notebook widget
5. KNN (k=5) classifier trained on labeled embeddings in PCA space
6. Classification is done at the **track level** — each track casts one vote per sampled embedding and the majority label wins, making it robust to brief ID swaps and noisy frames

Three files are saved per game to `output/classifiers/`:
- `{slug}_classifier.pkl` — fitted PCA + KNN model (loaded at inference time)
- `{slug}_labels.npz` — labeled embeddings + labels (for refitting); also stores track-level review data for the review widget
- `{slug}_pass1.pkl` — raw embeddings + best crops from Pass 1 (used by review and validation cells)

**Review widget** (`01_team_classification.ipynb`): after labeling, shows all crops grouped by assigned team so mislabeled outliers are immediately visible. Corrections can be saved and the classifier refit in one click.

**Validation**: a 2-minute annotated clip per game is saved to `output/classifier_validation/` for visual QC — watch the output and check that team colors are consistent before moving to event detection.

### 4. Pitch Homography (`src/run_pnlcalib_video.py`)

Maps image pixels to pitch coordinates (meters on a 105x68m FIFA-standard pitch) using **PnLCalib**, a pretrained HRNet-based encoder-decoder that outputs a 3x4 camera projection matrix.

**Post-processing pipeline (v2):**
1. **Non-gameplay filtering** — Skip replays/close-ups via scoreboard edge density check
2. **Player sanity check** — Reject projections where <50% of detected players land on pitch
3. **Line alignment check** — Verify projected lines match visible white pixels in the frame
4. **Temporal consistency** — Reject single-frame outliers; require 3 consecutive agreeing frames before accepting large projection changes
5. **EMA smoothing** — Blend consecutive projections (alpha=0.3) for stability

**Results across all 16 matches (60-second clips):**

| Tier | Matches | Coverage |
|------|---------|----------|
| Tier 1 (>80%) | jez-ars, dec-mla, jez-jed, sut-pet, sut-mla, mla-bud-2, jed-ars | 84-100% |
| Tier 2 (60-80%) | pet-mor, pet-bok, bok-jed, bud-sut | 64-75% |
| Tier 3 (<60%) | mor-bud, mor-ars, ars-dec | 10-43% |

Average: **71.3%** coverage. Tier 3 failures are caused by night/dusk lighting, oblique camera angles, and running track stadiums — conditions outside PnLCalib's training distribution.

```bash
python -m src.run_pnlcalib_video --match dec-mla --offset_min 10 --duration_sec 60 --alpha 0.3
```

**Per-frame fallback chain** (now wired into the runner): PnLCalib → sanity + line-alignment checks → optical-flow propagation (`camera_motion.py`, Lucas-Kanade on pitch features re-detected every 60 frames) → nearest manual seed (`manual_calibration.py`, `data/manual_calibration/{slug}_frame_{N:07d}.json`) → smoother stale-hold for up to 3 seconds. The end-of-run breakdown attributes each frame to one of these sources so it's immediately visible which stage is carrying the clip.

**Ground-truth widgets** for evaluating and seeding the homography (all share the same on-disk schema — `load_match_seeds` reads any of them):
- `build_labeling_widget` — click named landmarks (corners, halfway/CC intersections, PA + 6yd corners, penalty spots).
- `build_line_labeling_widget` — drag pitch lines and tap arc-intersection points; one unified solver handles both.
- `build_line_adjust_widget` — pre-projects every named line using the pipeline's current `P` so the user clicks to confirm and drags an endpoint to nudge — fastest path when the pipeline is mostly right.

### 5. Demo Video (`src/run_demo.py`, `notebooks/04_demo_video.ipynb`)

Polished clip renderer used as a pitch artefact: tracked players with team-coloured ellipses, name + jersey-number badges, live km/h + cumulative distance, ball triangle, pitch-line homography overlay, and a bottom-right minimap. Each per-game render needs `PLAYER_NAMES: {track_id: (name, team_id, jersey_no)}` mapped from a `--discover`-mode pass that draws raw track IDs.

The notebook bundles an iteration loop: scrub the rendered MP4 in a labeling widget tagging each frame Good / Drift / Wrong overlay / etc.; calibrate ground-truth `P` on the worst frames using the line-adjust widget seeded by the pipeline's own projection; and a compare cell that overlays current pipeline (white) vs ground truth (cyan) with per-frame mean pixel error at 7 reference world points. Saved GTs double as runtime seeds for the next render.

```bash
# Find track IDs first
python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16 --discover
# Edit PLAYER_NAMES, then render
python -m src.run_demo --match sut-mla --start_ts 1:04:48 --duration_sec 16
```

## Dataset

971 labeled frames across 16 Montenegro 1.CFL matches (822 train / 149 val, match-level split). Annotations done in CVAT, exported as YOLO format. Validation split is at match level (entire matches held out) to prevent data leakage.

## Tech Stack

- Python 3.10+, Windows 11, NVIDIA RTX 5070 (CUDA 12.8)
- **YOLOv8m** (Ultralytics) — fine-tuned for 2-class detection (person, ball)
- **BoT-SORT** — multi-object tracking
- **YOLOv8n-seg** — instance segmentation for team classification
- **ResNet18** (frozen ImageNet backbone) — player appearance embeddings
- **PCA + KNN** (scikit-learn) — supervised team classification (k=5, labeled per game)
- **PnLCalib** (HRNetV2-W48) — pretrained pitch calibration model
- **easyocr** — match clock reading
- **OpenCV** — video I/O, image processing, homography estimation

## Quick Start

```bash
pip install -r requirements.txt

# Per-game team classification labeling + review + validation
jupyter notebook notebooks/01_team_classification.ipynb

# Run homography on a match clip
python -m src.run_pnlcalib_video --match dec-mla --offset_min 10 --duration_sec 60
```

## Known Issues

- **Team classification** all 16 matches labeled; struggles on ~6 where jersey colors are similar or lighting is difficult (night games, oblique cameras). Majority vote per track reduces label-switch errors but ID swaps from long occlusions can still cause systematic misclassification.
- **Referee/GK filtering** is weak — refs often cluster with one team. Future fix: use pitch position via homography
- **Pitch homography** averages 71% coverage on raw PnLCalib v2; night games and oblique cameras remain challenging. Optical-flow propagation and manual seeds are wired in as fallbacks but the per-frame drift / wrong-overlay rate on demo clips is still high — actively being diagnosed via the GT iteration loop in `04_demo_video.ipynb`.
- **ars-dec source video** has a ~3:14 recording gap at the start of the second half
