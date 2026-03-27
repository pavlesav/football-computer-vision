import cv2
import numpy as np
import shutil
from pathlib import Path

from .config import Config
from .detection import PlayerDetector


def extract_dataset_frames(
    video_path: Path = None,
    output_dir: Path = None,
    max_frames: int = 300,
) -> Path:
    """
    Extract frames evenly distributed across the entire video for labeling.

    Args:
        video_path: Source video
        output_dir: Where to save the dataset (default: project/dataset/)
        max_frames: Number of frames to extract (evenly spaced across the video)

    Returns:
        Path to the dataset directory
    """
    video_path = video_path or Config.FULL_MATCH_VIDEO
    output_dir = output_dir or Config.PROJECT_ROOT / "dataset"

    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(max_frames, total)
    # Compute evenly spaced frame indices across the whole video
    frame_indices = np.linspace(0, total - 1, n_frames, dtype=int)

    print(f"Extracting {n_frames} frames evenly across {video_path.name}")
    print(f"  Total video frames: {total}")
    print(f"  Interval: ~every {total // n_frames} frames")

    saved = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"frame_{int(frame_idx):06d}.jpg"
        cv2.imwrite(str(images_dir / filename), frame)
        saved += 1

    cap.release()
    print(f"  Extracted {saved} frames → {images_dir}")
    return output_dir


# YOLO custom class mapping — 2-class model
# Role assignment (team A/B, goalkeeper, referee) is handled in post-processing.
CUSTOM_CLASSES = {
    0: "person",
    1: "ball",
}


def pre_annotate(
    dataset_dir: Path = None,
    detector: PlayerDetector = None,
) -> int:
    """
    Pre-annotate extracted frames using current YOLO model.

    Generates YOLO-format .txt label files as a starting point.
    You then review and correct these in a labeling tool.

    YOLO format per line: class_id x_center y_center width height
    (all values normalized to 0-1)

    Returns:
        Number of frames annotated
    """
    dataset_dir = dataset_dir or Config.PROJECT_ROOT / "dataset"
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = PlayerDetector()

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print("No images found. Run extract_dataset_frames() first.")
        return 0

    print(f"Pre-annotating {len(image_files)} images...")
    total_players = 0
    total_balls = 0

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]

        detections = detector.detect(frame)

        label_path = labels_dir / img_path.with_suffix(".txt").name
        lines = []

        for p in detections["players"]:
            x1, y1, x2, y2 = p["bbox"]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # Class 0 = player
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
            total_players += 1

        for b in detections["ball"]:
            x1, y1, x2, y2 = b["bbox"]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # Class 1 = ball
            lines.append(f"1 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
            total_balls += 1

        label_path.write_text("\n".join(lines))

    print(f"  Pre-annotated: {total_players} players, {total_balls} balls")
    print(f"  Labels saved → {labels_dir}")
    return len(image_files)


def export_cvat_zip(dataset_dir: Path = None) -> Path:
    """
    Package pre-annotations into a zip that CVAT can import (YOLO 1.1 format).

    CVAT requires:
      - obj.data    (class count + names file reference)
      - obj.names   (one class name per line)
      - train.txt   (list of image filenames)
      - obj_train_data/  (all .txt label files)
    """
    import zipfile

    dataset_dir = dataset_dir or Config.PROJECT_ROOT / "dataset"
    labels_dir = dataset_dir / "labels" / "train"
    images_dir = dataset_dir / "images" / "train"
    zip_path = dataset_dir / "cvat_annotations.zip"

    label_files = sorted(labels_dir.glob("*.txt"))
    image_files = sorted(images_dir.glob("*.jpg"))

    class_names = ["person", "ball"]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # obj.names — one class per line
        obj_names = "\n".join(class_names)
        zf.writestr("obj.names", obj_names)

        # obj.data
        obj_data = f"classes = {len(class_names)}\nnames = obj.names\ntrain = train.txt\n"
        zf.writestr("obj.data", obj_data)

        # train.txt — list of image filenames
        train_txt = "\n".join(f"obj_train_data/{img.name}" for img in image_files)
        zf.writestr("train.txt", train_txt)

        # Label files inside obj_train_data/
        for lbl in label_files:
            zf.write(lbl, f"obj_train_data/{lbl.name}")

    print(f"CVAT zip created: {zip_path}")
    print(f"  {len(label_files)} label files, {len(class_names)} classes")
    print(f"  Upload this in CVAT → Menu → Upload annotations → YOLO 1.1")
    return zip_path


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


def split_val(dataset_dir: Path = None, val_ratio: float = 0.2):
    """
    Split images+labels into train/val. Safe to call multiple times —
    always merges everything back into train first, then re-splits.
    """
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
    n_val = max(1, int(len(all_images) * val_ratio))

    # Deterministic shuffle
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_images))
    val_indices = indices[:n_val]

    moved = 0
    for idx in val_indices:
        img = all_images[idx]
        lbl = train_lbls / img.with_suffix(".txt").name

        shutil.move(str(img), str(val_imgs / img.name))
        if lbl.exists():
            shutil.move(str(lbl), str(val_lbls / lbl.name))
        moved += 1

    total = len(all_images)
    print(f"Split: {total - moved} train, {moved} val (from {total} total, {val_ratio:.0%} split)")


def visualize_labels(dataset_dir: Path = None, n_samples: int = 6):
    """
    Show a grid of images with their YOLO label boxes overlaid.
    Useful for verifying pre-annotations before manual review.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dataset_dir = dataset_dir or Config.PROJECT_ROOT / "dataset"
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print("No images found.")
        return

    # Evenly sample across the dataset
    step = max(1, len(image_files) // n_samples)
    samples = image_files[::step][:n_samples]

    colors = {0: "lime", 1: "cyan", 2: "orange", 3: "gray"}
    class_names = CUSTOM_CLASSES

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

                # Convert normalized to pixel coords
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
                    x1, y1 - 3, class_names.get(cls_id, f"cls{cls_id}"),
                    color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                )

    # Hide unused axes
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # Hide unused axes
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Pre-annotations — Review these for correctness", fontsize=14)
    plt.tight_layout()
    plt.show()
