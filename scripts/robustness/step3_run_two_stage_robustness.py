"""
Step 3: Run two-stage pipeline (YOLO + CNN) on clean test and on each corrupted test set; save metrics to CSV.

Calls step3_two_stage_inference then step4_evaluate_two_stage for each condition.
Metrics collected: Detection F1, End-to-end F1, Classification accuracy.

Run from project root (after step1):
  python3 scripts/robustness/step3_run_two_stage_robustness.py

Notes:
- This script reuses two-stage baseline scripts:
  - step3_two_stage_inference.py (writes predictions_{split}_{suffix}.json)
  - step4_evaluate_two_stage.py (prints detection F1, end-to-end F1, cls accuracy)
- Metrics are parsed from Step 4 stdout so all conditions are evaluated with
  the same matching logic used in the main two-stage pipeline.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

# Project root anchor so all subprocesses run with stable relative paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Input root produced by Step 1 (clean + corruption folders).
CORRUPTED_ROOT = PROJECT_ROOT / "data" / "processed_corrupted"
# Reference scripts from the two-stage baseline pipeline.
STEP3 = PROJECT_ROOT / "scripts" / "two_stage_baseline" / "step3_two_stage_inference.py"
STEP4 = PROJECT_ROOT / "scripts" / "two_stage_baseline" / "step4_evaluate_two_stage.py"
# Output CSV consumed by Step 4 robustness reporting.
OUT_CSV = PROJECT_ROOT / "runs" / "robustness" / "two_stage_robustness_metrics.csv"


def run_step4_and_parse(suffix: str) -> dict:
    """Run step4 with --split test and optional --suffix; parse stdout for Detection F1, E2E F1, Accuracy."""
    # Reference: reuse the project's canonical two-stage evaluation logic.
    # This avoids metric drift from re-implementing matching rules here.
    cmd = [sys.executable, str(STEP4), "--split", "test"]
    if suffix:
        cmd += ["--suffix", suffix]
    # Capture stdout so we can parse metrics into a machine-readable CSV.
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        return {}
    # Parse: first "F1:" = detection F1, second "F1:" = e2e F1, then "Accuracy:"
    lines = result.stdout.splitlines()
    f1_det = None
    f1_e2e = None
    acc = None
    f1_count = 0
    for line in lines:
        m = re.match(r"\s*F1:\s+([\d.]+)", line)
        if m:
            f1_count += 1
            if f1_count == 1:
                f1_det = float(m.group(1))
            elif f1_count == 2:
                f1_e2e = float(m.group(1))
        m = re.match(r"\s*Accuracy:\s+([\d.]+)", line)
        if m:
            acc = float(m.group(1))
    return {"detection_f1": f1_det, "e2e_f1": f1_e2e, "cls_accuracy": acc}


def main():
    # Minimal CLI to allow swapping classifier checkpoint if needed.
    parser = argparse.ArgumentParser(description="Run two-stage on clean + corrupted test sets")
    parser.add_argument("--classifier", type=str, default=None,
                        help="Classifier weights (default: runs/classifier_27k_finetuned/best.pt)")
    parser.add_argument("--out_csv", type=Path, default=OUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    # Default classifier is fine-tuned checkpoint used in dissertation robustness runs.
    classifier = args.classifier or str(PROJECT_ROOT / "runs" / "classifier_27k_finetuned" / "best.pt")
    # Detector remains fixed to Condition D YOLO checkpoint for fair comparison.
    yolo_weights = PROJECT_ROOT / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
    if not yolo_weights.exists():
        print(f"YOLO weights not found: {yolo_weights}")
        return
    if not Path(classifier).exists():
        print(f"Classifier not found: {classifier}")
        return

    # Collect one metrics row per condition.
    rows = []
    # Loop over clean + all corrupted (step1 creates clean/ and each folder has images/test)
    if not CORRUPTED_ROOT.exists():
        print(f"Corrupted root not found: {CORRUPTED_ROOT}. Run step1 first.")
        return
    for d in sorted(CORRUPTED_ROOT.iterdir()):
        if not d.is_dir():
            continue
        # Each condition folder from Step 1 exposes images/test for inference.
        img_dir = d / "images" / "test"
        if not img_dir.exists():
            continue
        name = d.name
        print(f"Running two-stage on {name}...")
        # Run two-stage inference and save predictions with condition suffix.
        subprocess.run(
            [
                sys.executable, str(STEP3), "--split", "test",
                "--images_dir", str(img_dir), "--suffix", name,
                "--classifier_weights", classifier,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        # Evaluate predictions with canonical Step 4 script and parse its metrics.
        m = run_step4_and_parse(name)
        if m:
            rows.append({"condition": name, **m})
            print("  detection_f1:", m["detection_f1"], "e2e_f1:", m["e2e_f1"], "cls_accuracy:", m["cls_accuracy"])

    # Write machine-readable CSV for downstream drop analysis in robustness Step 4.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "detection_f1", "e2e_f1", "cls_accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.out_csv}")
    print("Next: python3 scripts/robustness/step4_report_robustness.py")


if __name__ == "__main__":
    main()
