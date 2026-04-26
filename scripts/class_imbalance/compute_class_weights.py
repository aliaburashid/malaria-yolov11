"""
Compute class weights for imbalanced training labels (parasitized vs uninfected).

Formula: weight_c = total_cells / (2 * n_c), then normalized so weights sum to 2.
Rarer class gets higher weight; used in scripts/class_imbalance/train.py (WeightedDetectionLoss).

References:
- sklearn.utils.class_weight.compute_class_weight 'balanced': inverse frequency
  (n_samples / (n_classes * n_samples_per_class)); same idea as here.
  https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
- He & Garcia, "Learning from Imbalanced Data", IEEE TKDE 2009: survey of
  resampling and cost-sensitive learning; inverse frequency is a standard choice.
  https://ieeexplore.ieee.org/document/5128907
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def count_classes():
    """
    Count class 0 (parasitized) and class 1 (uninfected)
    across all YOLO label files in the training set.
    """
    # Initialize counters for both classes
    n0 = n1 = 0
    # Loop over all .txt label files in the train folder
    for p in LABELS_DIR.glob("*.txt"):
        with open(p) as f: # Open each label file
            for line in f: # Read each line in the label file
                line = line.strip() # Remove leading/trailing whitespace
                if not line: # Skip empty lines
                    continue

                # YOLO format: class_id x_center y_center width height
                # We only need the first value (class_id)
                cid = line.split()[0]

                # If class is 0 (parasitized), increment counter
                if cid == "0":
                    n0 += 1

                # If class is 1 (uninfected), increment counter
                elif cid == "1":
                    n1 += 1
    return n0, n1 # Return total counts for both classes


def compute_weights(n0: int, n1: int):
    """
    Compute class weights so that the rarer class receives higher weight.
    """
    # Total number of labeled cells
    total = n0 + n1

    # If one class is missing entirely, return equal weights (prevents division by zero)
    if n0 == 0 or n1 == 0:
        return [1.0, 1.0]
    # Inverse-frequency weighting: weight_c = total / (2 * n_c). Same idea as
    # sklearn 'balanced' and https://ieeexplore.ieee.org/document/5128907. Rare class -> larger weight.
    w0 = total / (2 * n0)  # parasitized
    w1 = total / (2 * n1)  # uninfected

    # Normalize weights so they sum to number of classes (2)
    # This keeps loss scale stable during training
    s = w0 + w1
    w0, w1 = 2 * w0 / s, 2 * w1 / s
    # Round weights to 4 decimal places for cleaner config output
    return [round(w0, 4), round(w1, 4)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute class weights from train labels")
    parser.add_argument("--write", action="store_true", help="Update config/default.yaml with computed weights")
    args = parser.parse_args()

    if not LABELS_DIR.exists():
        print(f"Labels dir not found: {LABELS_DIR}")
        print("Run create_splits.py and convert_to_yolo.py first.")
        return

    # Count class instances
    n0, n1 = count_classes()
    # Compute class weights
    weights = compute_weights(n0, n1)

    print(f"Train labels: parasitized={n0}, uninfected={n1}") # Print class counts
    print(f"Class weights (parasitized, uninfected): {weights}") # Print computed weights
    print("Add to config/default.yaml: class_weights:", weights)  # Tell user what to add in config file

    if args.write:
        text = CONFIG_PATH.read_text()
        old = "class_weights: null"
        new = f"class_weights: [{weights[0]}, {weights[1]}]"
        if old not in text:
            print("Config has no 'class_weights: null'; add class_weights manually:", weights)
        else:
            CONFIG_PATH.write_text(text.replace(old, new, 1))
            print(f"Updated {CONFIG_PATH}")


if __name__ == "__main__":
    main()
