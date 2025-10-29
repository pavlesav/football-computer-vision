# ⚽ Football Team Detection & Tracking

Automatic team detection and ball tracking for football/soccer videos using computer vision and deep learning.

![Demo](demo/demo_screenshot.png)

## 🎯 Features

- **Automatic Team Clustering**: Uses ResNet50 embeddings + UMAP + K-Means to automatically identify teams by uniform color
- **Real-time Player Tracking**: Tracks players across frames with temporal smoothing
- **Smart Ball Detection**: Possession-based ball tracking with motion prediction
- **Multi-Video Support**: Optimized for both short clips and full matches
- **No Manual Labeling**: Fully unsupervised - just provide the video!

## 🚀 Demo

| Input | Output |
|-------|--------|
| Raw football footage | Annotated with team colors + ball tracking |

**Features in Action:**
- 🔵 Blue ellipses = Team 1
- ⚪ Gray ellipses = Team 2  
- 🟡 Yellow ellipses = Referee
- 🟢 Green triangle = Ball position

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- scikit-learn
- umap-learn
- transformers (optional, for SigLip)

## 🎬 Quick Start

### 1. Short Videos (< 1 minute)

```python
# Open code.ipynb in Jupyter
# Set VIDEO_PATH = 'your_video.mp4'
# Run all cells
```

**Processing time**: ~2-5 minutes on CPU for 10-second clip

### 2. Long Videos (7+ minutes)

```python
# Open 7_min_test.ipynb in Jupyter
# Uses ResNet50 (10-15x faster) + frame sampling
# Set VIDEO_PATH = 'your_long_video.mp4'
# Run all cells
```

**Processing time**: ~5-10 minutes on CPU for 7-minute video

## 📊 How It Works

### Pipeline Overview

