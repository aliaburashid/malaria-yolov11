"""
Step 2: YOLO robustness metrics aligned with the two-stage greedy IoU protocol.

For each folder under data/processed_corrupted/ (clean + corruptions), this script:
  1) Runs the detector on that folder's test images with Ultralytics predict().
  2) Writes a Step-3-compatible predictions JSON (same keys as step3_two_stage_inference).
  3) Invokes step4_evaluate_two_stage.py on that JSON and parses stdout for F1 / accuracy.

That matches how scripts/robustness/step3_run_two_stage_robustness.py scores the two-stage
pipeline (subprocess to Step 4), so YOLO and two-stage robustness tables use the same
matching rule: greedy one-to-one IoU >= 0.5, class-agnostic localisation match, then
class check for end-to-end and matched-label accuracy (see step4_evaluate_two_stage.py).

Legacy note: older versions of this file used model.val() mAP-style metrics; those are
not comparable to Step 4 end-to-end F1. Re-run this script after Option A.

Run from project root (after step1):
  python3 scripts/robustness/step2_run_yolo_robustness.py
  python3 scripts/robustness/step2_run_yolo_robustness.py --detector_weights runs/detect/malaria/weights/best.pt

References:
- Ultralytics predict API (source of predict() usage):
  https://docs.ultralytics.com/modes/predict/
- Ultralytics YOLO class:
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py
"""

# Postpone annotation evaluation so forward references in type hints work on older Python.
from __future__ import annotations

# Stdlib: CLI, tabular output, JSON predictions, regex parsing of Step 4 stdout, subprocess runner.
import argparse
import csv
import json
import re
import subprocess
import sys
# Paths resolved relative to repo root regardless of cwd.
from pathlib import Path

# NumPy for detector tensor → Python lists; PIL loads RGB arrays for predict().
import numpy as np
from PIL import Image

# Repo root: this file is scripts/robustness/, so parents[2] is the repository root.
repo_root = Path(__file__).resolve().parent.parent.parent

# Corrupted-test image trees produced by step1_create_corrupted_test_sets.py.
corruption_root = repo_root / "data" / "processed_corrupted"

# Default weights: Condition D (oversampled + weighted), same default as two-stage robustness step3.
default_detector_weights = (
    repo_root / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
)

# Cached prediction JSONs (one file per corruption condition) for debugging / reuse.
default_predictions_dir = repo_root / "runs" / "robustness" / "yolo_greedy_predictions"

# Canonical greedy IoU evaluator shared with the two-stage baseline scripts.
step4_evaluator = repo_root / "scripts" / "two_stage_baseline" / "step4_evaluate_two_stage.py"

# Output table consumed by step4_report_robustness.py (column names match two_stage_robustness_metrics.csv).
output_metrics_csv = repo_root / "runs" / "robustness" / "yolo_robustness_metrics.csv"


def parse_step4_stdout(stdout_text: str) -> dict[str, float | None]:
    """
    Pull detection F1, end-to-end F1, and matched-label accuracy from Step 4 printout.

    Step 4 always prints two lines containing 'F1:' (detection block first, then
    end-to-end). Order is relied on by scripts/robustness/step3_run_two_stage_robustness.py
    as well; keep parsers in sync if Step 4 wording changes.
    """
    # Parsed floats; None if the corresponding pattern never matched.
    detection_f1: float | None = None
    end_to_end_f1: float | None = None
    matched_label_accuracy: float | None = None
    # Track occurrence index of "F1:" lines (first = detection, second = e2e).
    f1_line_index = 0
    # Walk stdout line-by-line; order matches Step 4 print layout.
    for line in stdout_text.splitlines():
        # First block of metrics uses "F1:" for detection-only performance.
        f1_match = re.match(r"\s*F1:\s+([\d.]+)", line)
        # Assign first hit to detection F1, second hit to end-to-end F1.
        if f1_match:
            f1_line_index += 1
            value = float(f1_match.group(1))
            if f1_line_index == 1:
                detection_f1 = value
            elif f1_line_index == 2:
                end_to_end_f1 = value
        # Matched-crop classifier accuracy line (single occurrence).
        acc_match = re.match(r"\s*Accuracy:\s+([\d.]+)", line)
        if acc_match:
            matched_label_accuracy = float(acc_match.group(1))
    # Return dict keys aligned with CSV columns downstream.
    return {
        "detection_f1": detection_f1,
        "e2e_f1": end_to_end_f1,
        "cls_accuracy": matched_label_accuracy,
    }


def write_predictions_json_for_folder(
    detector_weights: Path,
    test_images_dir: Path,
    label_split_name: str,
    confidence_cutoff: float,
    inference_size: int,
    destination_json: Path,
) -> None:
    """
    Run YOLO on every image in test_images_dir and write Step-3-style JSON.

    JSON layout matches step3_two_stage_inference.py output: top-level 'split',
    'images' map with per-image 'dets' list of dicts with xyxy, cls, det_conf.

    Ultralytics usage follows their predict() examples (single image numpy array as source).
    """
    # Fail fast if checkpoint path is wrong before loading the model.
    if not detector_weights.exists():
        raise FileNotFoundError(f"Detector checkpoint missing: {detector_weights}")
    # Require a directory before globbing images.
    if not test_images_dir.is_dir():
        raise FileNotFoundError(f"Test image folder missing: {test_images_dir}")

    # Stable order: all JPGs then all PNGs (sorted within each extension).
    image_paths = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    # Nothing to run if folder is empty or wrong layout.
    if not image_paths:
        raise FileNotFoundError(f"No jpg/png images under {test_images_dir}")

    # Lazy import keeps CLI help fast when ultralytics is not needed.
    from ultralytics import YOLO  # type: ignore  # ultralytics package

    # Load detector once; reused for every image in this corruption folder.
    detector = YOLO(str(detector_weights))
    # Skeleton matching Step 3 JSON: split name + empty images dict filled in the loop.
    payload: dict = {"split": label_split_name, "images": {}}

    # Run inference per image and accumulate serializable detection dicts.
    for index, image_path in enumerate(image_paths):
        # Ultralytics expects RGB HWC uint8/float array for numpy source.
        rgb_array = np.array(Image.open(image_path).convert("RGB"))
        # predict() API: https://docs.ultralytics.com/modes/predict/
        inference_result = detector.predict(
            source=rgb_array,
            conf=confidence_cutoff,
            imgsz=inference_size,
            verbose=False,
        )[0]

        # Default empty list if model finds no boxes above conf threshold.
        detection_list: list[dict] = []
        # Boxes may be None when there are zero detections.
        if inference_result.boxes is not None:
            # Tensors on device; move to CPU numpy for JSON serialization.
            boxes_xyxy = inference_result.boxes.xyxy
            boxes_conf = inference_result.boxes.conf
            boxes_cls = inference_result.boxes.cls
            # Each row is one detection.
            for box_index in range(boxes_xyxy.shape[0]):
                detection_list.append(
                    {
                        "xyxy": boxes_xyxy[box_index].cpu().numpy().tolist(),
                        "det_conf": float(boxes_conf[box_index].cpu().item()),
                        "cls": int(boxes_cls[box_index].cpu().item()),
                    }
                )

        # Prefer repo-relative path keys so Step 4 can join to label files.
        try:
            relative_key = str(image_path.relative_to(repo_root))
        # If image lives outside repo_root, fall back to basename only.
        except ValueError:
            relative_key = image_path.name
        # One entry per image filename key.
        payload["images"][relative_key] = {"dets": detection_list}

        # Heartbeat so long runs show progress in the terminal.
        if (index + 1) % 50 == 0 or (index + 1) == len(image_paths):
            print(f"    {index + 1}/{len(image_paths)} images")

    # Provenance + hyperparameters for reproducibility inside the JSON artifact.
    payload["meta"] = {
        "eval_protocol": "yolo_predictions_plus_step4_greedy_iou",
        "detector_weights": str(detector_weights),
        "confidence": confidence_cutoff,
        "inference_size": inference_size,
        "test_images_dir": str(test_images_dir),
    }

    # Ensure output directory exists before writing.
    destination_json.parent.mkdir(parents=True, exist_ok=True)
    # Indented JSON for logs and version control diffs.
    with open(destination_json, "w") as handle:
        json.dump(payload, handle, indent=2)


def run_step4_on_predictions(prediction_json: Path, iou_match_threshold: float) -> dict[str, float | None]:
    """Subprocess Step 4; same pattern as step3_run_two_stage_robustness.run_step4_and_parse."""
    # Invoke same Python interpreter + Step 4 script with explicit predictions path.
    command = [
        sys.executable,
        str(step4_evaluator),
        "--predictions",
        str(prediction_json),
        "--iou",
        str(iou_match_threshold),
    ]
    # Run from repo root so relative paths inside Step 4 resolve consistently.
    completed = subprocess.run(command, cwd=str(repo_root), capture_output=True, text=True)
    # Non-zero exit: print diagnostics and return sentinel Nones for CSV skip logic.
    if completed.returncode != 0:
        print(completed.stderr or completed.stdout, file=sys.stderr)
        return {"detection_f1": None, "e2e_f1": None, "cls_accuracy": None}
    # Parse structured metrics from Step 4 stdout on success.
    return parse_step4_stdout(completed.stdout)


def main() -> None:
    # Describe script purpose in --help.
    parser = argparse.ArgumentParser(
        description="YOLO greedy-IoU robustness metrics (predict + Step 4), aligned with two-stage CSV."
    )
    # Detector checkpoint; aliases match older flag names in notes/scripts.
    parser.add_argument(
        "--detector_weights",
        "--yolo_weights",
        type=Path,
        default=default_detector_weights,
        dest="detector_weights",
        help="Path to YOLO weights (default: Condition D best.pt). Alias: --yolo_weights",
    )
    # Score threshold passed through to Ultralytics predict(conf=...).
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for keeping detections (match crowded-field / Step 3 default)",
    )
    # Letterboxed resize side length for the detector.
    parser.add_argument(
        "--inference_size",
        type=int,
        default=640,
        help="Square input size passed to predict(imgsz=...)",
    )
    # Greedy matcher IoU cutoff passed through to Step 4 (keep aligned with main evaluation defaults).
    parser.add_argument(
        "--iou_match",
        type=float,
        default=0.5,
        help="IoU threshold forwarded to Step 4 matcher (same default as step4_evaluate_two_stage.IOU_THRESH)",
    )
    # Where per-condition prediction JSON files are written.
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=default_predictions_dir,
        help="Directory to store per-condition prediction JSON files",
    )
    # Final aggregated metrics table path for Step 4 reporter.
    parser.add_argument("--out_csv", type=Path, default=output_metrics_csv, help="Output CSV path")
    # Populate Namespace from argv.
    arguments = parser.parse_args()

    # Abort early if default/custom weights path is invalid.
    if not arguments.detector_weights.exists():
        print(f"Detector weights not found: {arguments.detector_weights}")
        return

    # Step 1 must have produced corruption folders under this root.
    if not corruption_root.exists():
        print(f"Corruption dataset root not found: {corruption_root}. Run step1 first.")
        return

    # Accumulator for CSV rows (one dict per corruption condition successfully evaluated).
    table_rows: list[dict[str, str | float]] = []
    # Alphabetical folder iteration for deterministic CSV row order.
    condition_dirs = sorted(p for p in corruption_root.iterdir() if p.is_dir())

    # Outer loop: one corruption condition at a time.
    for condition_path in condition_dirs:
        # Step 1 layout: <condition>/images/test holds PNG/JPG tiles.
        test_folder = condition_path / "images" / "test"
        # Skip malformed folders without a test split.
        if not test_folder.is_dir():
            continue

        # Folder basename becomes JSON suffix and CSV condition column.
        condition_slug = condition_path.name
        # Unique predictions file per condition for caching and debugging.
        prediction_path = arguments.predictions_dir / f"predictions_test_{condition_slug}.json"

        print(f"{condition_slug}: running detector + Step 4...")
        try:
            # Generate predictions JSON then evaluate with Step 4 below.
            write_predictions_json_for_folder(
                detector_weights=arguments.detector_weights,
                test_images_dir=test_folder,
                label_split_name="test",
                confidence_cutoff=arguments.confidence,
                inference_size=arguments.inference_size,
                destination_json=prediction_path,
            )
        # Missing images or weights raised inside writer — skip this condition cleanly.
        except FileNotFoundError as exc:
            print(f"  skip: {exc}")
            continue

        # Run greedy matcher + metrics extraction via subprocess.
        metrics = run_step4_on_predictions(prediction_path, arguments.iou_match)
        # Step 4 failure leaves e2e_f1 unset — omit row from CSV.
        if metrics["e2e_f1"] is None:
            print(f"  skip: Step 4 failed for {condition_slug}")
            continue

        # Round floats for stable CSV diffs.
        table_rows.append(
            {
                "condition": condition_slug,
                "detection_f1": round(float(metrics["detection_f1"] or 0.0), 6),
                "e2e_f1": round(float(metrics["e2e_f1"] or 0.0), 6),
                "cls_accuracy": round(float(metrics["cls_accuracy"] or 0.0), 6),
            }
        )
        # Console summary mirrors CSV columns for quick sanity check.
        print(
            f"  e2e_f1={metrics['e2e_f1']:.4f}  detection_f1={metrics['detection_f1']:.4f}  "
            f"cls_accuracy={metrics['cls_accuracy']:.4f}"
        )

    # Nothing succeeded — avoid writing an empty misleading CSV.
    if not table_rows:
        print("No rows written; check corruption folders and image paths.")
        return

    # Prepare output directory for metrics CSV.
    arguments.out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write header + one row per condition evaluated successfully.
    with open(arguments.out_csv, "w", newline="") as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=["condition", "detection_f1", "e2e_f1", "cls_accuracy"],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    # Confirm path on stdout for the next manual step in the robustness pipeline.
    print(f"Saved: {arguments.out_csv}")
    print("Next: python3 scripts/robustness/step3_run_two_stage_robustness.py")


# Entry point when executed as a script (not when imported).
if __name__ == "__main__":
    main()
