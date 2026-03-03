"""
Build oversampled training list for Condition C (data-level oversampling).

Purpose: Increase the frequency of images that contain at least one parasitized
(class 0) cell during training.

Strategy:
If an image contains ≥1 parasitized cell → repeat it multiple times 
in the training list (.txt file).

Validation and test sets are NOT modified.
Source / references:
- Condition C: data-level counterpart to Condition B (loss weighting).
- Ultralytics accepts train as a .txt file (one image path per line); paths relative to path in dataset yaml.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "train_oversampled.txt"

# Number of extra times to repeat parasitized images
# Example:
#   2 -> image appears 3× total (1 original + 2 extra)
#   3 -> image appears 4× total
PARASITIZED_EXTRA_REPEATS = 2  # so each parasitized image appears 3×

# Check if a label file contains at least one parasitized cell (class 0)

def has_parasitized(label_path: Path) -> bool:
    """
    Returns True if the label file contains at least one object
    with class ID 0 (parasitized).
    """
    with open(label_path) as f: # Open label file
        for line in f:  # Read each annotation line
            line = line.strip() # Remove leading/trailing whitespace
            if not line: # Skip empty lines
                continue
            if line.split()[0] == "0": # Check if class_id == "0"
                return True  # Found parasitized cell
    return False  # If no class 0 found


def main():
    # Import argument parser to allow optional command-line arguments
    import argparse
    # Create parser with description
    parser = argparse.ArgumentParser(description="Build oversampled train list for Condition C")
    parser.add_argument("--repeats", type=int, default=PARASITIZED_EXTRA_REPEATS, help=f"Extra repeats for parasitized images (default {PARASITIZED_EXTRA_REPEATS} = 3× total)")
    args = parser.parse_args()

    if not LABELS_DIR.exists():
        print(f"Labels dir not found: {LABELS_DIR}")
        print("Run create_splits.py and convert_to_yolo.py first.")
        return

    # Use absolute paths to avoid resolution issues when train list is a .txt file
    train_lines = [] # List to store final training image paths (with oversampling)
    # Counters for reporting
    n_parasitized = 0
    n_uninfected_only = 0

    # Loop through all label files in sorted order
    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        stem = label_path.stem # Extract filename without extension
        # Construct corresponding image path (.jpg assumed)
        img_path = IMAGES_DIR / f"{stem}.jpg"
        # Skip if image file does not exist
        if not img_path.exists():
            continue
        img_str = str(img_path.resolve())
        is_parasitized = has_parasitized(label_path)
        if is_parasitized:
            n_parasitized += 1
            # Add once + extra repeats
            train_lines.append(img_str)
            for _ in range(args.repeats):
                train_lines.append(img_str)
        else:
            n_uninfected_only += 1
            train_lines.append(img_str)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(train_lines) + "\n")

    n_total = len(train_lines)
    print(f"Train labels: {n_parasitized} images with parasitized, {n_uninfected_only} uninfected-only")
    print(f"Oversampled list: {n_total} lines (parasitized images ×{args.repeats + 1})")
    print(f"Written: {OUTPUT_FILE}")
    print("Use config/dataset_oversampled.yaml and train with: python3 scripts/train.py --oversample")


if __name__ == "__main__":
    main()
