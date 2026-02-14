"""
Convert NIH Polygon Set annotations to YOLO format.
- Reads polygon boundaries and converts to bounding boxes
- Outputs normalized YOLO format: class_id x_center y_center width height
- Splits by patient (uses create_splits output)

Source / references:
- NIH-NLM Thin Blood Smear (Pf) dataset: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
- YOLO label format: https://docs.ultralytics.com/datasets/detect/
"""

import csv
import shutil
from pathlib import Path

# Paths - NIH dataset is sibling folder to malaria-yolov11
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIH_POLYGON_PATH = PROJECT_ROOT.parent / "NIH-NLM-ThinBloodSmearsPf" / "Polygon Set"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Class mapping (RBC only; White_Blood_Cell skipped)
CLASS_MAP = {"Parasitized": 0, "Uninfected": 1}


def load_split(split_name):
    """Load patient IDs for a split."""
    path = SPLITS_DIR / f"{split_name}_patients.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Run create_splits.py first. Missing: {path}"
        )
    with open(path) as f:
        reader = csv.DictReader(f)
        return [row["patient_id"] for row in reader]


def parse_polygon_line(line, img_width, img_height):
    """Parse one annotation line. Returns (class_id, x_min, y_min, x_max, y_max) or None."""
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None

    cell_type = parts[1].strip()
    if cell_type not in CLASS_MAP:
        return None  # Skip White_Blood_Cell and others

    anno_type = parts[3].strip()
    if anno_type != "Polygon":
        return None

    class_id = CLASS_MAP[cell_type]
    num_points = int(parts[4])
    coords = [float(x) for x in parts[5 : 5 + num_points * 2]]

    if len(coords) < 4:
        return None

    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Clamp to image bounds
    x_min = max(0, min(x_min, img_width - 1))
    x_max = max(0, min(x_max, img_width))
    y_min = max(0, min(y_min, img_height - 1))
    y_max = max(0, min(y_max, img_height))

    if x_max <= x_min or y_max <= y_min:
        return None

    return (class_id, x_min, y_min, x_max, y_max, img_width, img_height)


def box_to_yolo(class_id, x_min, y_min, x_max, y_max, img_width, img_height):
    """Convert bbox to YOLO normalized format."""
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_patient(patient_id, split_name, images_dir, labels_dir):
    """Convert all images for one patient."""
    patient_path = NIH_POLYGON_PATH / patient_id
    img_path = patient_path / "Img"
    gt_path = patient_path / "GT"

    if not img_path.exists() or not gt_path.exists():
        return 0, 0

    n_images = 0
    n_cells = 0

    for img_file in img_path.glob("*.jpg"):
        gt_file = gt_path / (img_file.stem + ".txt")
        if not gt_file.exists():
            continue

        with open(gt_file) as f:
            lines = f.readlines()

        if len(lines) < 2:
            continue

        # First line: num_cells, width, height
        first = lines[0].strip().split(",")
        img_width = float(first[1])
        img_height = float(first[2])

        yolo_lines = []
        for line in lines[1:]:
            result = parse_polygon_line(line, img_width, img_height)
            if result:
                class_id, x_min, y_min, x_max, y_max, w, h = result
                yolo_lines.append(box_to_yolo(class_id, x_min, y_min, x_max, y_max, w, h))

        if len(yolo_lines) == 0:
            continue

        # Unique filename: patient_id + image name
        out_name = f"{patient_id}_{img_file.name}"
        out_img = images_dir / out_name
        out_label = labels_dir / (out_name.replace(".jpg", ".txt"))

        shutil.copy(img_file, out_img)
        with open(out_label, "w") as f:
            f.write("\n".join(yolo_lines))

        n_images += 1
        n_cells += len(yolo_lines)

    return n_images, n_cells


def convert_all():
    """Convert all splits."""
    if not NIH_POLYGON_PATH.exists():
        raise FileNotFoundError(
            f"NIH Polygon Set not found at {NIH_POLYGON_PATH}"
        )

    for split_name in ["train", "val", "test"]:
        patient_ids = load_split(split_name)
        images_dir = OUTPUT_DIR / "images" / split_name
        labels_dir = OUTPUT_DIR / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        total_imgs = 0
        total_cells = 0
        for pid in patient_ids:
            n_imgs, n_cells = convert_patient(pid, split_name, images_dir, labels_dir)
            total_imgs += n_imgs
            total_cells += n_cells

        print(f"  {split_name}: {total_imgs} images, {total_cells} cells")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Converting NIH Polygon Set to YOLO format...")
    convert_all()
    print("Done.")
