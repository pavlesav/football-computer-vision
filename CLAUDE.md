# Football Computer Vision - Project Guide

## What This Project Does

Automated analysis of broadcast football (soccer) footage from the Montenegro First Division (1.CFL). The pipeline takes a full-match video and produces annotated output with:
- Player and ball detection (bounding boxes)
- Persistent player tracking across frames (track IDs)
- Team classification (Team A vs Team B) via jersey appearance
- Ball position interpolation to fill detection gaps
- Annotated video output with team-colored ellipses and ball markers

## Tech Stack

- **Python 3.10+** on Windows 11, NVIDIA RTX 5070 (sm_120, CUDA 12.8)
- **YOLOv8m** (Ultralytics) - fine-tuned for 2-class detection (person, ball)
- **BoT-SORT** - multi-object tracking via Ultralytics' built-in tracker
- **YOLOv8n-seg** - instance segmentation to mask player silhouettes (background removal)
- **ResNet18** (frozen ImageNet backbone) - 512-dim CNN embeddings from masked player crops
- **PCA + GMM** (scikit-learn) - dimensionality reduction (512->32) then 2-component Gaussian Mixture Model for team clustering
- **OpenCV** - video I/O, image processing
- **Plotly + ipywidgets** - interactive 3D cluster visualization in notebooks

## Project Structure

```
football-computer-vision/
├── main.ipynb                     # Main pipeline: detection, tracking, team classification, export
├── model_fine_tuning.ipynb        # YOLO training + error analysis
├── prepare_frames_for_annotation.ipynb  # Dataset QA: summary stats + visual label browser
├── pitch_homography.ipynb         # WIP: pitch line detection & camera calibration
├── src/
│   ├── __init__.py                # Lazy imports
│   ├── config.py                  # All paths, thresholds, hyperparameters
│   ├── detection.py               # PlayerDetector (YOLO), BallInterpolator
│   ├── segmentation.py            # YOLOv8-seg masks, ResNet18 CNN embeddings
│   ├── team_classifier.py         # TeamClassifier: PCA + GMM clustering
│   ├── tracking.py                # Tracker: track-level team assignment
│   ├── visualization.py           # Annotator: ellipses, triangles, bboxes
│   ├── video_utils.py             # Video I/O, clip extraction (ffmpeg)
│   └── dataset.py                 # Dataset management, train/val split
├── dataset/
│   ├── data.yaml                  # YOLO dataset config (2 classes)
│   ├── images/{all,train,val}/    # 971 annotated frames
│   ├── labels/{all,train,val}/    # YOLO-format labels (class cx cy w h)
│   └── cvat_exports/              # Archived CVAT annotation exports
├── runs/detect/                   # YOLO training outputs (weights, curves)
├── videos/                        # Source match videos (gitignored)
├── output/                        # Annotated output videos (gitignored)
├── yolov8m.pt                     # Base YOLO model
└── yolov8n-seg.pt                 # Segmentation model for team classification
```

## Data Flow (main.ipynb batch pipeline)

The batch annotation cell processes each game in two passes:

```
Pass 1: Detect + Track + Sample Embeddings
  For every frame:
    YOLO detection + BoT-SORT tracking → all_detections[]
  Every 15th frame:
    YOLOv8-seg (full frame) → person masks
    Masked crops → ResNet18 → 512-dim embeddings
    Store per-track: track_embeddings[track_id].append(embedding)

Between passes: Fit & Classify
  All embeddings → PCA (512→32) → GMM (k=2) fit
  Per track: mean(embeddings) → GMM predict → team_id
  BallInterpolator fills gaps up to 15 frames

Pass 2: Annotate
  Re-read video, draw team-colored ellipses + ball triangles → MP4
```

**Key design choice:** Team classification is done at the **track level**, not per-frame. Each track's embeddings are averaged into one vector and classified once. This eliminates flickering and is more robust than per-frame classification with majority voting.

## Detection Model

- **Architecture:** YOLOv8m (25M params)
- **Classes:** 2 - `person` (0), `ball` (1)
- **Training:** 150 epochs, imgsz=1280, batch=8, mosaic=1.0, copy_paste=0.3, patience=30
- **Dataset:** 971 frames across 16 matches (822 train / 149 val, match-level split)
- **Best model:** `runs/detect/football_2class_yolov8m_v22/weights/best.pt`
- **Metrics:** mAP50=0.944, mAP50-95=0.748, Precision=0.952, Recall=0.892
- **Weight resolution:** `Config.resolve_yolo_model()` auto-finds the latest `runs/detect/*/weights/best.pt`

Roles like goalkeeper, referee, and team assignment are NOT part of detection - they're handled entirely in post-processing (team classifier + future position heuristics).

## Team Classification Pipeline

1. **Segmentation:** YOLOv8n-seg runs on the full frame (not crops!) to produce person silhouette masks
2. **Masking:** Each player's crop is masked - background pixels replaced with mean foreground color to avoid CNN bias
3. **Embedding:** Frozen ResNet18 backbone produces 512-dim vectors per masked crop, resized to 128x64 (tall format for full-body players)
4. **PCA:** 512 -> 32 dims, removes noise
5. **GMM fit:** Two-pass process:
   - Pass 1: Rough k=2 GMM on all embeddings
   - Pass 2: Remove bottom 5% by membership probability, re-fit for tighter centroids
6. **Track-level prediction:** Average all embeddings for a track, classify the mean embedding once

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

## Known Limitations & Active Work

- **Team classification** works well on ~10/16 matches, struggles on ~6 where jersey colors are similar or lighting is difficult
- **Referee/GK filtering** via bottom-5% GMM membership probability is weak - refs often cluster tightly with one team. Future plan: use pitch position (requires homography)
- **Pitch homography** is in progress (`pitch_homography.ipynb`) - classical line detection + RANSAC. Will enable: minimap projection, offside lines, GK detection by position
- **Ball tracking** uses simple linear interpolation across gaps. Could be improved with tracker-aware logic

## Conventions

- All image processing uses **BGR** (OpenCV default). Convert to RGB only for display (matplotlib, PIL).
- Video timestamps in the pipeline typically use `10:00-15:00` (minute 10-15 of the broadcast) to skip pre-game footage.
- `src/` modules use relative imports (`from .config import Config`). Notebooks add `sys.path.insert(0, str(Path.cwd()))` and use `importlib.reload()` for hot-reloading during development.
- The 16 matches are all from Montenegro's 1.CFL (First Football League), seasons 25/26.
