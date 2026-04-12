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
| Team Classification | Done | ResNet18 + PCA + GMM, works on ~10/16 matches |
| Pitch Homography | In Progress | Classical line detection, works on ~40% of frames |
| Event Detection | Planned | Requires homography |

## Project Structure

```
football-computer-vision/
├── main.ipynb                 # Main pipeline: detection, tracking, team classification, export
├── model_fine_tuning.ipynb    # YOLO training + error analysis
├── src/
│   ├── __init__.py            # Lazy imports
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
│   ├── data.yaml              # YOLO dataset config (2 classes: person, ball)
│   ├── images/{all,train,val}/  # 971 annotated frames across 16 matches
│   ├── labels/{all,train,val}/  # YOLO-format labels
│   └── cvat_exports/          # Archived CVAT annotation exports
├── runs/detect/               # YOLO training outputs (weights, curves)
├── videos/                    # Source match videos (gitignored)
├── output/                    # Annotated output videos (gitignored)
├── period_detection_fast_results.json  # Cached period boundaries for all 16 matches
├── requirements.txt
└── README.md
```

## Layer Details

### 1. Broadcast Analysis (`src/broadcast.py`)

Fully unsupervised segmentation of a broadcast into game periods.

- **Camera cut detection**: HSV histogram difference between consecutive frames (threshold 0.4). Cut count varies widely by production (4 cuts for single-camera to 586 for multi-camera).
- **Period detection**: OCR of the match clock via easyocr on 60 uniformly-sampled frames. Extrapolates kickoff times from clock readings: `clock < 40:00` maps to first half, `clock >= 50:00` maps to second half. The 40-50 minute range is ambiguous and skipped.
- **Validation**: Sanity checks (both halves detected, 45-75 min gap, reasonable positions in broadcast). Tested on all 16 matches: 15 PASS, 1 WARN (ars-dec has a source video recording gap).

```python
from src.broadcast import detect_period_starts, find_game_periods
```

### 2. Detection + Tracking (`src/detection.py`)

- **YOLOv8m** fine-tuned on 971 frames across 16 matches (2 classes: person, ball)
- **BoT-SORT** tracking provides persistent IDs across frames
- **Ball interpolation** fills detection gaps up to 15 frames via linear interpolation
- Metrics: mAP50=0.944, mAP50-95=0.748, Precision=0.952, Recall=0.892

### 3. Team Classification (`src/team_classifier.py`, `src/segmentation.py`)

1. YOLOv8n-seg produces per-player silhouette masks on the full frame
2. Masked crops (background removed) fed to frozen ResNet18 for 512-dim embeddings
3. PCA (512->32 dims) + 2-component GMM clusters players into teams
4. Classification is done at the **track level** (average all embeddings per track, classify once) to eliminate flickering

### 4. Pitch Homography (`src/homography.py`)

Maps image pixels to pitch coordinates (meters on a 105x68m FIFA-standard pitch). Classical computer vision pipeline:

1. **Field segmentation**: HSV green mask to isolate the pitch
2. **White line detection**: Dual-channel approach:
   - Color-based: bright + low saturation pixels
   - CLAHE-enhanced white top-hat transform for structural line detection
   - Player bounding boxes masked out via YOLO to eliminate white jersey false positives
   - Adaptive thresholds for day vs night games
3. **Line detection + merging**: Probabilistic Hough Transform, then co-linear segments merged
4. **Semantic classification**: Lines classified as touchline (horizontal), perpendicular (steep), or other (intermediate angle)
5. **Touchline identification**: Near (bottom of frame) and far (top) touchlines identified by y-position
6. **Perpendicular line identification**: Spacing patterns matched against known FIFA pitch geometry (PA edge at 16.5m, halfway at 52.5m, etc.)
7. **Homography estimation**: Detected keypoints (line intersections) matched to pitch template coordinates, RANSAC-based homography with player projection sanity check

```python
from src.homography import estimate_homography
result = estimate_homography(frame, player_boxes=yolo_boxes)
if result.H is not None:
    # result.H is 3x3 matrix mapping image pixels to pitch meters
    mapped = cv2.perspectiveTransform(pixel_pts, result.H)
```

**Current results** (tested on 5 matches x 3 frames):
- 6/15 frames produce a working homography (~40%)
- All successful homographies have < 1.5m reprojection error
- Best results on daylight games with clear pitch markings (sut-mla: 2/3, dec-mla: 2/3)

**Known failure modes:**
- Insufficient perpendicular lines — midfield views without penalty area structure lack x-coordinate anchors
- Wrong line identification — spacing heuristic can misidentify which pitch line a detected perpendicular is
- Poor pitch condition — worn markings, unusual stadiums (e.g., running tracks between camera and pitch)

**Highest-impact improvements planned:**
- Center circle detection via ellipse fitting — visible in most frames, would anchor pitch center
- Hypothesis testing for line identification — try multiple assignments, score by player projection quality
- Temporal smoothing — propagate homography from nearby frames to fill gaps

## Dataset

971 labeled frames across 16 Montenegro 1.CFL matches (822 train / 149 val, match-level split):

| Match | Slug | Frames |
|-------|------|--------|
| Buducnost vs Sutjeska (R20) | bud-sut | 300 |
| Arsenal vs Decic (R22) | ars-dec | 60 |
| Bokelj vs Jedinstvo (R22) | bok-jed | 60 |
| Jezero vs Jedinstvo (R22) | jez-jed | 60 |
| Petrovac vs Mornar (R22) | pet-mor | 60 |
| Sutjeska vs Mladost (R22) | sut-mla | 60 |
| + 10 additional matches | various | 371 |

Annotations done in CVAT, exported as YOLO format. Validation split is at match level (entire matches held out) to prevent data leakage.

## Tech Stack

- Python 3.10+, Windows 11, NVIDIA RTX 5070 (CUDA 12.8)
- YOLOv8m (Ultralytics) for detection, YOLOv8n-seg for segmentation
- BoT-SORT for multi-object tracking
- ResNet18 (frozen ImageNet backbone) for player appearance embeddings
- PCA + GMM (scikit-learn) for team clustering
- easyocr for match clock reading
- OpenCV for image processing and homography estimation

## Quick Start

```bash
pip install -r requirements.txt
ffmpeg -version  # required for clip extraction

# Run main pipeline
jupyter notebook main.ipynb
```

## Known Issues

- **Team classification** struggles on ~6/16 matches where jersey colors are similar or lighting is difficult
- **Referee/GK filtering** is weak — refs often cluster with one team. Future fix: use pitch position via homography
- **ars-dec source video** has a ~3:14 recording gap at the start of the second half (match clock starts at 48:14). This is a camera issue, not a detection bug.
- **Pitch homography** works on ~40% of frames — needs center circle detection and temporal smoothing for production use
