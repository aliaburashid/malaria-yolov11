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

# PROJECT_ROOT points to the main project folder (malaria-yolov11).
# __file__ is this script, then .parent.parent moves up to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# NIH_POLYGON_PATH points to the original NIH dataset folder.
# The script assumes the NIH folder is stored next to the project folder.
# "Polygon Set" contains one folder per patient.
NIH_POLYGON_PATH = PROJECT_ROOT.parent / "NIH-NLM-ThinBloodSmearsPf" / "Polygon Set"

# SPLITS_DIR is where the patient split CSV files were saved by create_splits.py.
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# OUTPUT_DIR is where the YOLO-ready dataset will be written.
# This folder will contain images/ and labels/ subfolders for train/val/test.
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Class mapping (RBC only; White_Blood_Cell skipped)
# CLASS_MAP maps NIH class names to YOLO class IDs.
CLASS_MAP = {"Parasitized": 0, "Uninfected": 1}


def load_split(split_name):
    """Load patient IDs for a split.""" 
    # loads a CSV file that lists patient IDs
    # belonging to the requested split (train/val/test).
    path = SPLITS_DIR / f"{split_name}_patients.csv"

     # If the split file does not exist, the pipeline cannot continue.
    #  forcing the user to run create_splits.py first.
    if not path.exists():
        raise FileNotFoundError(
            f"Run create_splits.py first. Missing: {path}"
        )
    
    # The CSV contains a column named "patient_id".
    # The function returns a list of those patient IDs.
    with open(path) as f:
        reader = csv.DictReader(f)
        return [row["patient_id"] for row in reader]


def parse_polygon_line(line, img_width, img_height):
    """
    Parses one annotation row describing one cell.
    - If it is a polygon for Parasitized/Uninfected, it is converted to a bounding box.
    - If it is not relevant (e.g., White_Blood_Cell or non-polygon), it is skipped.
    """
    # NIH annotation files are comma-separated.
    parts = line.strip().split(",")
    # If the line is too short, it cannot contain a valid polygon definition.
    if len(parts) < 6:
        return None
    
    # The second column contains the cell type (Parasitized/Uninfected).
    cell_type = parts[1].strip()
    # If the cell type is not in CLASS_MAP, it is ignored.
    # This skips White_Blood_Cell and any other labels.
    if cell_type not in CLASS_MAP:
        return None  

    # The fourth column contains the annotation type (Polygon).
    anno_type = parts[3].strip()
    # If the annotation type is not a polygon, it is ignored.
    if anno_type != "Polygon":
        return None

    # Convert the cell type into a YOLO class ID (0 or 1).
    class_id = CLASS_MAP[cell_type]
    # The next value is how many polygon points exist.
    num_points = int(parts[4])
    # After that, the file provides coordinates:
    # x1, y1, x2, y2, ... for num_points points.
    coords = [float(x) for x in parts[5 : 5 + num_points * 2]]

    # If there are not at least 4 coordinates, the polygon is invalid.
    if len(coords) < 4:
        return None

    # The coordinates are split into x and y values.
    xs = coords[0::2]
    ys = coords[1::2]
    
    # The x and y values are used to calculate the bounding box.
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Clamp bounding box values to image bounds.
    # This prevents boxes from going outside the image (which would break YOLO training).
    x_min = max(0, min(x_min, img_width - 1))
    x_max = max(0, min(x_max, img_width))
    y_min = max(0, min(y_min, img_height - 1))
    y_max = max(0, min(y_max, img_height))

    # If the box is invalid (negative or zero area), it is ignored.
    if x_max <= x_min or y_max <= y_min:
        return None

    # The function returns the class ID, bounding box coordinates, and image dimensions.
    return (class_id, x_min, y_min, x_max, y_max, img_width, img_height)


def box_to_yolo(class_id, x_min, y_min, x_max, y_max, img_width, img_height):
    """ 
    Convert bounding box to YOLO normalized format.
    YOLO expects bbox coordinates as: center_x, center_y, width, height all normalized to 0–1.
    This function converts from pixel coordinates into that format.
    """

    # Compute bbox center in pixels, then divide by image width/height to normalize.
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    # Compute bbox width/height in pixels, then normalize.
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    # Return the bbox in YOLO format.
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_patient(patient_id, split_name, images_dir, labels_dir):
    """
    Convert all images for one patient.
    processes one patient folder at a time.
    - For each image, the matching annotation file is read.
    - Polygons are converted to YOLO boxes and written to a .txt label file.
    - The image is copied into the correct split folder.
    """
     # Patient folder path (e.g., Polygon Set/<patient_id>/)
    patient_path = NIH_POLYGON_PATH / patient_id

    # NIH structure uses Img/ for images and GT/ for annotation files.
    img_path = patient_path / "Img"
    gt_path = patient_path / "GT"

    # If this patient folder is incomplete, skip it safely.
    if not img_path.exists() or not gt_path.exists():
        return 0, 0

    # Track how many images and cell boxes are converted for logging.
    n_images = 0
    n_cells = 0

    # Loop through every .jpg image for this patient.
    for img_file in img_path.glob("*.jpg"):
        # Find the matching annotation file (e.g., 247C99P60ThinF.jpg -> 247C99P60ThinF.txt)
        gt_file = gt_path / (img_file.stem + ".txt")
        if not gt_file.exists():
            continue
        
        # Read all annotation lines.
        with open(gt_file) as f:
            lines = f.readlines()
        
        # If there are not at least 2 lines, the file is invalid.
        if len(lines) < 2:
            continue

        # The first line contains metadata including image width and height.
        first = lines[0].strip().split(",")
        img_width = float(first[1])
        img_height = float(first[2])

        # Convert every polygon annotation line into YOLO format.
        yolo_lines = []
        for line in lines[1:]:
            result = parse_polygon_line(line, img_width, img_height)
            if result:
                class_id, x_min, y_min, x_max, y_max, w, h = result
                yolo_lines.append(box_to_yolo(class_id, x_min, y_min, x_max, y_max, w, h))

        # If no valid RBC boxes exist (e.g., only White_Blood_Cell), skip the image.
        if len(yolo_lines) == 0:
            continue

        # Unique filename: patient_id + image name
        # This prevents collisions if different patients have the same image name.
        out_name = f"{patient_id}_{img_file.name}"
        # Output image path for this split.
        out_img = images_dir / out_name
        # Output label path must match image name (same stem) with .txt extension.
        out_label = labels_dir / (out_name.replace(".jpg", ".txt"))

        # Copy the image into the processed dataset folder.
        # This ensures the image is in the correct location for training.
        shutil.copy(img_file, out_img)
        # Write YOLO label file with one line per bounding box.
        with open(out_label, "w") as f:
            f.write("\n".join(yolo_lines))

        # Increment counters for logging.
        n_images += 1
        n_cells += len(yolo_lines)

    # Return how many images and boxes were produced for this patient.
    return n_images, n_cells


def convert_all():
    """Convert all splits."""

    # Ensure the NIH dataset path exists.
    if not NIH_POLYGON_PATH.exists():
        raise FileNotFoundError(
            f"NIH Polygon Set not found at {NIH_POLYGON_PATH}"
        )

    # Loop through each split (train, val, test).   
    for split_name in ["train", "val", "test"]:
        # Load the patient IDs for this split.
        patient_ids = load_split(split_name)
        # Create output directories for this split.
        images_dir = OUTPUT_DIR / "images" / split_name
        labels_dir = OUTPUT_DIR / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Track totals for this split.
        total_imgs = 0
        total_cells = 0
        # Convert each patient in the split.
        for pid in patient_ids:
            n_imgs, n_cells = convert_patient(pid, split_name, images_dir, labels_dir)
            total_imgs += n_imgs
            total_cells += n_cells

        # Print summary for this split.
        print(f"  {split_name}: {total_imgs} images, {total_cells} cells")
    # Print final output location for the full dataset.
    print(f"\nOutput: {OUTPUT_DIR}")

# Execute only when run directly
if __name__ == "__main__":
    print("Converting NIH Polygon Set to YOLO format...")
    convert_all()
    print("Done.")
