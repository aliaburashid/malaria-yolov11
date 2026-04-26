"""
Step 2: Run YOLO on clean test set and on each corrupted test set; save metrics to CSV.

Metrics collected: Precision, Recall, mAP50, mAP50-95 (overall).
Uses one YOLO weights file (default: best from Condition D).

Run from project root (after step1):
  python3 scripts/robustness/step2_run_yolo_robustness.py
  python3 scripts/robustness/step2_run_yolo_robustness.py --yolo_weights runs/detect/malaria/weights/best.pt

Source:
- Ultralytics validation API (model.val) for P/R/mAP metrics:
  https://github.com/ultralytics/ultralytics
"""

import argparse
import csv
from pathlib import Path

# Project root anchor so paths are stable regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Input root produced by Step 1 (clean + corruption folders).
CORRUPTED_ROOT = PROJECT_ROOT / "data" / "processed_corrupted"
# Default detector is Condition D best checkpoint (main experiment setting).
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
# Output CSV consumed by Step 4 reporting.
OUT_CSV = PROJECT_ROOT / "runs" / "robustness" / "yolo_robustness_metrics.csv"


def run_yolo_val(weights_path: Path, data_yaml: Path, split: str = "test"):
    # Reference: Ultralytics YOLO validation API (model.val).
    # Docs/source: https://github.com/ultralytics/ultralytics
    from ultralytics import YOLO
    # Load detector checkpoint.
    model = YOLO(str(weights_path))
    # Run validation on requested split using dissertation image size/batch.
    metrics = model.val(data=str(data_yaml), split=split, imgsz=640, batch=8, verbose=False)
    # Ultralytics returns [P, R, mAP50, mAP50-95] via mean_results().
    overall = metrics.box.mean_results()
    p, r, m50, m5095 = overall[0], overall[1], overall[2], overall[3]
    # Compute F1 from P/R to keep one summary metric per condition.
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"P": p, "R": r, "F1": f1, "mAP50": m50, "mAP50-95": m5095}


def main():
    # CLI is kept minimal so this step is easy to rerun with alternative weights.
    parser = argparse.ArgumentParser(description="Run YOLO on clean + corrupted test sets")
    parser.add_argument("--yolo_weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to YOLO best.pt")
    parser.add_argument("--out_csv", type=Path, default=OUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    # Stop early if chosen checkpoint is missing.
    if not args.yolo_weights.exists():
        print(f"YOLO weights not found: {args.yolo_weights}")
        return

    # Collect one metrics row per condition folder.
    rows = []
    # Loop over clean + all corrupted (step1 creates clean/ first, then dataset.yaml in each folder)
    if not CORRUPTED_ROOT.exists():
        print(f"Corrupted root not found: {CORRUPTED_ROOT}. Run step1 first.")
        return
    # Deterministic folder order keeps output table stable across runs.
    dirs = sorted(CORRUPTED_ROOT.iterdir())
    if not any(d.is_dir() and (d / "dataset.yaml").exists() for d in dirs):
        print("No dataset.yaml found in any subfolder. Run step1 first.")
        return
    clean_dir = CORRUPTED_ROOT / "clean"
    if not (clean_dir.is_dir() and (clean_dir / "dataset.yaml").exists()):
        print("WARNING: clean/ folder with dataset.yaml not found. Step1 should create it first.")
        print("Run: python3 scripts/robustness/step1_create_corrupted_test_sets.py")
    for d in dirs:
        if not d.is_dir():
            continue
        # Each condition must provide its own dataset.yaml created by Step 1.
        yaml_path = d / "dataset.yaml"
        if not yaml_path.exists():
            continue
        # Evaluate this specific clean/corrupted condition on test split.
        m = run_yolo_val(args.yolo_weights, yaml_path, split="test")
        # Save condition name with metrics so Step 4 can compute drops vs clean.
        rows.append({"condition": d.name, **m})
        print(d.name, ":", m["F1"], "mAP50", m["mAP50"])

    # Write machine-readable CSV for reporting scripts and dissertation tables.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "P", "R", "F1", "mAP50", "mAP50-95"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.out_csv}")
    print("Next: python3 scripts/robustness/step3_run_two_stage_robustness.py")


if __name__ == "__main__":
    main()
