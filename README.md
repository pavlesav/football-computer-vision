# Football Match Computer Vision

Automated player detection, tracking, and team classification for broadcast football (soccer) footage.

## What it does

- **Player & Ball Detection** — YOLOv8 fine-tuned on 2 classes (`person`, `ball`); goalkeeper/referee roles assigned in post-processing
- **Object Tracking** — BoT-SORT assigns persistent IDs across frames
- **Team Classification** — YOLOv8-seg masks isolate jersey pixels → Lab median color → 2-component GMM clusters teams; low-confidence predictions flagged as Referee/GK
- **Temporal Smoothing** — Majority vote over track history prevents team assignment flickering
- **Annotated Output** — Exports video with team-colored ellipses under players and triangle markers on the ball

## Dataset

Montenegro First Division (1.CFL), Round 20 — **FK Budućnost vs FK Sutjeska**, 21.02.2026.

- 300 frames extracted and labeled in CVAT
- 2 YOLO classes: `person`, `ball` (goalkeeper/referee/team assigned in post-processing)
- Fine-tuned YOLOv8n: mAP50 = 0.741
- Ball annotations expanded with 60 frames each from 5 additional 1.CFL matches (Arsenal–Dečić, Bokelj–Jedinstvo, Jezero–Jedinstvo, Petrovac–Mornar, Sutjeska–Mladost)

## Project Structure

```
├── main.ipynb              # Main pipeline notebook
├── dataset_creation.ipynb  # Dataset extraction & labeling pipeline
├── src/
│   ├── config.py           # All configuration constants
│   ├── video_utils.py      # Video I/O, clip extraction (ffmpeg)
│   ├── detection.py        # YOLOv8 player & ball detection + BoT-SORT tracking
│   ├── segmentation.py     # YOLOv8-seg masks & Lab feature extraction
│   ├── team_classifier.py  # Seg-masked Lab median → GMM team assignment
│   ├── tracking.py         # Track management & temporal smoothing
│   ├── visualization.py    # Ellipses, triangles, bounding boxes
│   └── dataset.py          # Frame extraction & pre-annotation utilities
├── dataset/                # Training data (images gitignored)
│   ├── data.yaml           # YOLO dataset config
│   └── cvat_annotations.zip
├── videos/                 # Source videos (gitignored)
├── output/                 # Annotated output videos (gitignored)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure ffmpeg is available
ffmpeg -version

# Open and run the notebook
jupyter notebook main.ipynb
```

## Pipeline Overview

### 1. Detection (Fine-tuned YOLOv8 + BoT-SORT)
A YOLOv8n model fine-tuned on 300 labeled frames detects persons and the ball (2 classes). BoT-SORT provides persistent track IDs across frames.

### 2. Team Classification (YOLOv8-seg + Lab median + GMM)
For each detected person:
1. Run YOLOv8-seg on the full frame to produce per-player silhouette masks
2. Slice the jersey area (top 40% height, center 50% width) from the masked crop
3. Compute the median Lab color over only the masked foreground pixels — grass, ads, and background are excluded
4. A 2-component GMM fitted on sampled frames clusters players into Team A or Team B

Anyone whose max cluster-membership probability falls below the confidence threshold is classified as Referee/Goalkeeper — no separate Gaussian needed for outlier kits.

Lab is perceptually uniform: L (lightness) separates white from dark kits, a (green↔red) and b (blue↔yellow) separate most team colors. Running segmentation on the full frame (not crops) is critical for accurate silhouettes.

### 3. Temporal Smoothing
Each track's team assignment is smoothed using majority vote over its full history, preventing frame-to-frame flickering.

### 4. Visualization & Export
- **Players**: Team-colored ellipses at feet + track ID label
- **Ball**: Green triangle marker above the ball
- Output exported as MP4

## Requirements

- Python 3.10+
- ffmpeg (for clip extraction)
- CPU works for testing; GPU recommended for full match processing

## License

See [LICENSE](LICENSE).
