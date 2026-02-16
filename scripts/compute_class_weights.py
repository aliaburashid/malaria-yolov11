"""
Compute class weights for imbalanced train labels (parasitized vs uninfected).
Formula: weight_c = total_cells / (2 * n_c) so the rare class gets higher weight.
Run once and paste the output into config/default.yaml, or use --write to update the config.

Source / references:
- Inverse-frequency weighting: standard approach for imbalanced classification (e.g. sklearn balanced class_weight).
- Applied here to train-set label counts; used in scripts/train.py WeightedDetectionLoss.
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def count_classes():
    """Count class 0 (parasitized) and class 1 (uninfected) in train labels."""
    n0 = n1 = 0
    for p in LABELS_DIR.glob("*.txt"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid = line.split()[0]
                if cid == "0":
                    n0 += 1
                elif cid == "1":
                    n1 += 1
    return n0, n1


def compute_weights(n0: int, n1: int):
    """Weight so rare class (parasitized) has higher weight."""
    total = n0 + n1
    if n0 == 0 or n1 == 0:
        return [1.0, 1.0]
    w0 = total / (2 * n0)  # parasitized
    w1 = total / (2 * n1)  # uninfected
    # Normalize so weights sum to num_classes (2) - keeps loss scale similar
    s = w0 + w1
    w0, w1 = 2 * w0 / s, 2 * w1 / s
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

    n0, n1 = count_classes()
    weights = compute_weights(n0, n1)

    print(f"Train labels: parasitized={n0}, uninfected={n1}")
    print(f"Class weights (parasitized, uninfected): {weights}")
    print("Add to config/default.yaml: class_weights:", weights)

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
