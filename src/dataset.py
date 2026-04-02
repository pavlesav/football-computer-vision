import cv2
import numpy as np
import shutil
from pathlib import Path

from .config import Config


CLASSES = {0: "person", 1: "ball"}


def create_data_yaml(dataset_dir: Path = None) -> Path:
    """Create the data.yaml config file required for YOLO training."""
    dataset_dir = dataset_dir or Config.PROJECT_ROOT / "dataset"
    yaml_path = dataset_dir / "data.yaml"

    content = f"""# Football Detection Dataset
# 2-class model: YOLO detects person + ball only.
# Role assignment (team A/B, goalkeeper, referee) is handled in post-processing.

path: {dataset_dir.as_posix()}
train: images/train
val: images/val

nc: 2
names:
  0: person
  1: ball
"""
    yaml_path.write_text(content)
    print(f"Created {yaml_path}")
    return yaml_path


VAL_MATCHES = ["mla-bud-2", "jed-ars", "pet-mor"]


def split_val(dataset_dir: Path = None, val_matches: list = None):
    """
    Match-level train/val split. Safe to call multiple times —
    always merges everything back into train first, then re-splits.

    Frames whose filename starts with a val match slug go to val,
    everything else goes to train.
    """
    val_matches = val_matches or VAL_MATCHES
    dataset_dir = dataset_dir or Config.PROJECT_ROOT / "dataset"
    train_imgs = dataset_dir / "images" / "train"
    train_lbls = dataset_dir / "labels" / "train"
    val_imgs = dataset_dir / "images" / "val"
    val_lbls = dataset_dir / "labels" / "val"
    val_imgs.mkdir(parents=True, exist_ok=True)
    val_lbls.mkdir(parents=True, exist_ok=True)

    # Merge val back into train first (idempotent)
    for img in val_imgs.glob("*.jpg"):
        shutil.move(str(img), str(train_imgs / img.name))
    for lbl in val_lbls.glob("*.txt"):
        shutil.move(str(lbl), str(train_lbls / lbl.name))

    all_images = sorted(train_imgs.glob("*.jpg"))

    # Sort val slugs longest-first so "mla-bud-2" matches before "mla-bud"
    sorted_slugs = sorted(val_matches, key=len, reverse=True)

    def is_val(filename):
        for slug in sorted_slugs:
            if filename.startswith(slug + "_frame_"):
                return True
        return False

    moved = 0
    for img in all_images:
        if is_val(img.name):
            lbl = train_lbls / img.with_suffix(".txt").name
            shutil.move(str(img), str(val_imgs / img.name))
            if lbl.exists():
                shutil.move(str(lbl), str(val_lbls / lbl.name))
            moved += 1

    total = len(all_images)
    print(f"Split: {total - moved} train, {moved} val (from {total} total)")
    print(f"Val matches: {', '.join(val_matches)}")


def visualize_labels(dataset_dir: Path = None, n_samples: int = 6):
    """Show a grid of images with their YOLO label boxes overlaid."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dataset_dir = dataset_dir or Config.PROJECT_ROOT / "dataset"
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print("No images found.")
        return

    step = max(1, len(image_files) // n_samples)
    samples = image_files[::step][:n_samples]

    colors = {0: "lime", 1: "cyan"}

    cols = min(3, len(samples))
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, img_path in enumerate(samples):
        frame = cv2.imread(str(img_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        axes[i].imshow(frame_rgb)
        axes[i].set_title(img_path.name, fontsize=9)
        axes[i].axis("off")

        label_path = labels_dir / img_path.with_suffix(".txt").name
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                box_w = bw * w
                box_h = bh * h

                color = colors.get(cls_id, "white")
                rect = patches.Rectangle(
                    (x1, y1), box_w, box_h,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                axes[i].add_patch(rect)
                axes[i].text(
                    x1, y1 - 3, CLASSES.get(cls_id, f"cls{cls_id}"),
                    color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                )

    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Dataset labels — verify annotations", fontsize=14)
    plt.tight_layout()
    plt.show()
