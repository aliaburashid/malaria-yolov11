"""
Create patient-level train/val/test splits for NIH malaria dataset.

CRITICAL: Split by patient (not by image) to avoid data leakage: images from
the same patient must not appear in both train and val/test.

References:
- NIH-NLM Thin Blood Smear (P. falciparum) Polygon Set:
  https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
- Patient-level splitting: Rajaraman et al. PeerJ 2018; Kassim et al. IEEE JBHI 2021
- Reproducibility: fixed seed (config/default.yaml)
"""

import csv
# Ref: Python stdlib https://docs.python.org/3/library/csv.html
import random
# Ref: Python stdlib https://docs.python.org/3/library/random.html
from pathlib import Path
# Ref: Python stdlib https://docs.python.org/3/library/pathlib.html

# Fixed seed so the same split is produced every run (reproducibility)
# Ref: ML reproducibility; matches config/default.yaml seed
SEED = 42
random.seed(SEED)

# Where this script lives -> project root (two levels up from scripts/)
# Ref: pathlib resolve(), parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# NIH dataset folder: sibling of project; "Polygon Set" = cell annotations
# Ref: NIH download structure https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
NIH_POLYGON_PATH = PROJECT_ROOT.parent / "NIH-NLM-ThinBloodSmearsPf" / "Polygon Set"
# Output folder for the three CSV files (used by convert_to_yolo.py)
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Fraction of patients in each split; must sum to 1.0
# Ref: common convention; config/default.yaml data.train_ratio etc.; sklearn train_test_split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_patient_ids():
    """
    Get all patient folder IDs from NIH Polygon Set.
    Ref: NIH folder structure = one folder per patient.
    """
    # Check dataset exists before continuing
    if not NIH_POLYGON_PATH.exists():
        raise FileNotFoundError(
            f"NIH Polygon Set not found at {NIH_POLYGON_PATH}\n"
            "Ensure NIH-NLM-ThinBloodSmearsPf is in the project root."
        )
    # List dir names that are folders (each = one patient ID)
    # Ref: pathlib iterdir(), is_dir()
    patient_ids = [d.name for d in NIH_POLYGON_PATH.iterdir() if d.is_dir()]
    # Sort so order is deterministic before we shuffle in create_splits
    return sorted(patient_ids)


def create_splits():
    """
    Create train/val/test splits by patient. Shuffle list, then slice into three parts.
    Writes: train_patients.csv, val_patients.csv, test_patients.csv.
    """
    # Get list of all patient IDs from the dataset
    patient_ids = get_patient_ids()
    n_patients = len(patient_ids)

    if n_patients == 0:
        raise ValueError("No patient folders found.")

    # Shuffle in place; uses SEED so same split every time
    # Ref: random.shuffle https://docs.python.org/3/library/random.html#random.shuffle
    random.shuffle(patient_ids)

    # How many patients go in each split (train 70%, val 15%, test 15%)
    # Ref: sequential split (no stratification by infection status here)
    n_train = int(n_patients * TRAIN_RATIO)
    n_val = int(n_patients * VAL_RATIO)
    n_test = n_patients - n_train - n_val

    # Slice the shuffled list into three parts
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train : n_train + n_val]
    test_ids = patient_ids[n_train + n_val :]

    # Create data/splits/ if it does not exist
    # Ref: pathlib mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Write one CSV per split: header "patient_id", then one row per patient
    # Ref: csv.writer https://docs.python.org/3/library/csv.html#csv.writer
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


# Only run create_splits when this file is executed (not when imported)
# Ref: Python __main__ https://docs.python.org/3/library/__main__.html
if __name__ == "__main__":
    print("Creating patient-level splits...")
    create_splits()
    print("Done.")
