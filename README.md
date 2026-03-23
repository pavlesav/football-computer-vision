# Football Match Computer Vision

Automated player detection, tracking, and team classification for broadcast football (soccer) footage.

## What it does

- **Player & Ball Detection** — Fine-tuned YOLOv8n detects players, goalkeepers, referees, and the ball (4 custom classes)
- **Object Tracking** — ByteTrack assigns persistent IDs across frames
- **Team Classification** — ResNet18 appearance embeddings + KMeans clustering assigns each player to a team
- **Temporal Smoothing** — Majority vote over track history prevents team assignment flickering
- **Interactive Visualization** — 3D PCA scatter plot with Plotly for cluster introspection
- **Annotated Output** — Exports video with team-colored ellipses under players and triangle markers on the ball

## Dataset

Montenegro First Division (1.CFL), Round 20 — **FK Budućnost vs FK Sutjeska**, 21.02.2026.

- 300 frames extracted and labeled in CVAT
- 4 custom YOLO classes: `player`, `ball`, `goalkeeper`, `referee`
- Fine-tuned YOLOv8n: mAP50 = 0.741

## Project Structure

```
├── main.ipynb              # Main pipeline notebook
├── dataset_creation.ipynb  # Dataset extraction & labeling pipeline
├── src/
│   ├── config.py           # All configuration constants
│   ├── video_utils.py      # Video I/O, clip extraction (ffmpeg)
│   ├── detection.py        # YOLOv8 player & ball detection
│   ├── team_classifier.py  # ResNet18 embeddings → KMeans team assignment
│   ├── tracking.py         # ByteTrack wrapper & temporal smoothing
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

### 1. Detection (Fine-tuned YOLOv8 + ByteTrack)
A YOLOv8n model fine-tuned on 300 labeled frames detects players, goalkeepers, referees, and the ball. ByteTrack provides persistent track IDs across frames.

### 2. Team Classification (ResNet18 + KMeans)
For each detected player:
1. Crop the full bounding box
2. Extract a 512-dim appearance embedding via a pretrained ResNet18 backbone
3. A global KMeans classifier (fitted on sampled frames) assigns the embedding to Team A or Team B

YOLO class IDs route referees directly (no clustering needed).

### 3. Temporal Smoothing
Each track's team assignment is smoothed using majority vote over its history, preventing frame-to-frame flickering.

### 4. Visualization & Export
- **Players**: Team-colored ellipses at feet + track ID label
- **Ball**: Green triangle marker above the ball
- **3D PCA**: Interactive Plotly scatter plot to inspect cluster quality
- Output exported as MP4

## Requirements

- Python 3.10+
- ffmpeg (for clip extraction)
- CPU works for testing; GPU recommended for full match processing

## License

See [LICENSE](LICENSE).
