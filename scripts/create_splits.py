"""
Create patient-level train/val/test splits for NIH malaria dataset.
CRITICAL: Split by patient (not image) to avoid data leakage.

Source / references:
- NIH-NLM Thin Blood Smear (Pf) dataset: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
- Split logic: project-specific (patient-level to prevent leakage).
"""

import csv
import random
from pathlib import Path

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Paths - NIH dataset is sibling folder to malaria-yolov11
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIH_POLYGON_PATH = PROJECT_ROOT.parent / "NIH-NLM-ThinBloodSmearsPf" / "Polygon Set"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_patient_ids():
    """Get all patient folder IDs from Polygon Set."""
    if not NIH_POLYGON_PATH.exists():
        raise FileNotFoundError(
            f"NIH Polygon Set not found at {NIH_POLYGON_PATH}\n"
            "Ensure NIH-NLM-ThinBloodSmearsPf is in the project root."
        )
    patient_ids = [d.name for d in NIH_POLYGON_PATH.iterdir() if d.is_dir()]
    return sorted(patient_ids)


def create_splits():
    """Create stratified train/val/test splits by patient."""
    patient_ids = get_patient_ids()
    n_patients = len(patient_ids)

    if n_patients == 0:
        raise ValueError("No patient folders found.")

    # Shuffle with fixed seed
    random.shuffle(patient_ids)

    # Calculate split indices
    n_train = int(n_patients * TRAIN_RATIO)
    n_val = int(n_patients * VAL_RATIO)
    n_test = n_patients - n_train - n_val

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train : n_train + n_val]
    test_ids = patient_ids[n_train + n_val :]

    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = SPLITS_DIR / f"{split_name}_patients.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id"])
            for pid in ids:
                writer.writerow([pid])
        print(f"  {split_name}: {len(ids)} patients -> {path}")

    print(f"\nTotal: {n_patients} patients")
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    print("Creating patient-level splits...")
    create_splits()
    print("Done.")
