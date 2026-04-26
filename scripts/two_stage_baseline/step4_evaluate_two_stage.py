"""
Step 4: Evaluate the two-stage pipeline against ground truth.

Goal:
- Compare the predictions from Step 3 with the real labelled boxes (ground truth)
- Measure how well the two-stage pipeline performed

What this script checks:
1. Detection / localisation:
   Did the predicted box overlap the real box enough?
2. End-to-end correctness:
   Did the predicted box overlap the real box AND have the correct class?
3. Classification accuracy:
   For the boxes that matched, how often was the class correct?

Run from project root:
  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split val
  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split test
  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --predictions runs/two_stage_baseline/predictions_val.json
"""

# argparse lets us run the same script with options like --split val or --suffix finetuned
import argparse

# json is used to load the prediction file produced by Step 3
import json

# sys is used to stop the script early if something important is missing
import sys

# Path helps build safe file/folder paths
from pathlib import Path

# NumPy is used for box coordinates and IoU calculations
import numpy as np

# PIL is used to open the image and read its width/height
from PIL import Image

# ----------------------------
# Paths and constants
# ----------------------------

# Project root:
# This file is inside scripts/two_stage_baseline/, so go up 3 levels to repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default folder containing prediction JSON files from Step 3
PREDICTIONS_DIR = PROJECT_ROOT / "runs" / "two_stage_baseline"

# Root of processed dataset (contains images/ and labels/)
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

# IoU threshold used to decide whether a predicted box matches a ground-truth box
# Standard object detection threshold = 0.5
IOU_THRESH = 0.5

# Class names for readability
CLASS_NAMES = ["parasitized", "uninfected"]


def box_iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> float:
    """
    Compute IoU (Intersection over Union) between two boxes.

    What IoU means:
    - It measures how much two boxes overlap
    - 1.0 means perfect overlap
    - 0.0 means no overlap

    Why this matters:
    - In object detection, a predicted box is usually counted as correct
      only if it overlaps a real box enough (for example IoU >= 0.5)
    """

    # Find the overlapping rectangle between box A and box B
    x1 = max(a_xyxy[0], b_xyxy[0])
    y1 = max(a_xyxy[1], b_xyxy[1])
    x2 = min(a_xyxy[2], b_xyxy[2])
    y2 = min(a_xyxy[3], b_xyxy[3])

    # Width and height of the overlap region
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    # Area of overlap
    inter = inter_w * inter_h

    # Area of box A
    # max(0, ...) protects against invalid/reversed boxes
    area_a = max(0.0, a_xyxy[2] - a_xyxy[0]) * max(0.0, a_xyxy[3] - a_xyxy[1])

    # Area of box B
    area_b = max(0.0, b_xyxy[2] - b_xyxy[0]) * max(0.0, b_xyxy[3] - b_xyxy[1])

    # Union = total area covered by both boxes
    union = area_a + area_b - inter

    # IoU = overlap / union
    return inter / union if union > 0 else 0.0




def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """
    Load the ground-truth boxes from a YOLO label file.

    Label format in the file:
      class_id x_center y_center width height
    where all coordinates are normalized (between 0 and 1)

    This function converts them into:
      (class_id, [x1, y1, x2, y2]) in PIXELS

    Why:
    - Step 3 predictions are in pixel coordinates
    - so ground truth must also be converted to pixel coordinates
    - then both can be compared fairly
    """

    # If the label file does not exist, return no boxes
    if not label_path.exists():
        return []

    out = []

    with open(label_path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            parts = line.split()

            # Need at least 5 values in YOLO format
            if len(parts) < 5:
                continue

            # Read class id and normalized box values
            cls_id = int(parts[0])
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalized xywh -> pixel xyxy
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h

            # Clip to image boundaries
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Save one GT box as (class_id, xyxy)
            out.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float64)))

    return out


def match_predictions_to_gt(
    pred_xyxy_list: list, gt_boxes: list[tuple[int, np.ndarray]], iou_thresh: float
) -> list[tuple[int, int]]:
    """
    Match predicted boxes to ground-truth boxes using IoU.

    Output:
    - list of pairs: (pred_index, gt_index)

    Matching rule:
    - one predicted box can match only one GT box
    - one GT box can match only one predicted box
    - only matches with IoU >= threshold are allowed

    Why:
    - This is how we decide which predictions are true positives
    """

    # If there are no predictions or no GT boxes, nothing can match
    if not pred_xyxy_list or not gt_boxes:
        return []

    # Convert predictions and GT boxes into arrays
    pred_arr = np.array(pred_xyxy_list, dtype=np.float64)
    gt_xyxy = np.array([g[1] for g in gt_boxes], dtype=np.float64)

    # Number of predictions and number of GT boxes
    n_p, n_g = len(pred_arr), len(gt_xyxy)

    # IoU matrix:
    # rows = predictions
    # cols = GT boxes
    # ious[i, j] = IoU between prediction i and GT box j
    ious = np.zeros((n_p, n_g))

    # Fill in IoU matrix
    for i in range(n_p):
        for j in range(n_g):
            ious[i, j] = box_iou(pred_arr[i], gt_xyxy[j])

    # Collect all pairs that meet the IoU threshold
    candidates = []
    for i in range(n_p):
        for j in range(n_g):
            if ious[i, j] >= iou_thresh:
                candidates.append((ious[i, j], i, j))

    # Sort candidates by highest IoU first
    candidates.sort(key=lambda x: -x[0])

    # Track which predictions / GT boxes have already been used
    matched_p, matched_g = set(), set()
    result = []

    # Greedy one-to-one matching:
    # pick the highest-IoU pairs first, without reusing a box
    for _iou, pi, gi in candidates:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        result.append((pi, gi))

    return result


def main():
    # ----------------------------
    # Parse terminal arguments
    # ----------------------------

    parser = argparse.ArgumentParser(description="Evaluate two-stage pipeline vs GT")

    # Choose split-based file automatically:
    # predictions_val.json or predictions_test.json
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default=None,
        help="Use predictions_{split}.json (or with --suffix: predictions_{split}_{suffix}.json)"
    )

    # Optional: directly provide a predictions file path
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions JSON (overrides --split/--suffix/--predictions_dir)"
    )

    # Optional: change directory where predictions live
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Directory containing predictions JSON (default: runs/two_stage_baseline)"
    )

    # Optional suffix for files like predictions_val_finetuned.json
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix to match Step 3 output, e.g. 'finetuned' -> predictions_val_finetuned.json"
    )

    # IoU threshold for deciding a match
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESH,
        help="IoU threshold for match (default 0.5)"
    )

    args = parser.parse_args()

    # ----------------------------
    # Decide which prediction file to load
    # ----------------------------

    if args.predictions is not None:
        # If user directly gave a file path, use it
        pred_path = Path(args.predictions)

    elif args.split is not None:
        # Otherwise build file name from split + optional suffix
        name = f"predictions_{args.split}{'_' + args.suffix if args.suffix else ''}.json"
        pred_path = Path(args.predictions_dir) / name

    else:
        # Need at least one way to know which predictions to evaluate
        print("ERROR: Provide --split val|test or --predictions <path>")
        sys.exit(1)

    # Stop if predictions file does not exist
    if not pred_path.exists():
        print(f"ERROR: Predictions file not found: {pred_path}")
        print("Run Step 3 first: python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val")
        sys.exit(1)

    # ----------------------------
    # Load predictions JSON
    # ----------------------------

    with open(pred_path) as f:
        data = json.load(f)

    # Read which split this file belongs to
    split = data.get("split", args.split or "val")

    # Read all image entries
    images_data = data.get("images", {})

    # Stop if predictions JSON contains no images
    if not images_data:
        print("ERROR: No images in predictions JSON")
        sys.exit(1)

    # Ground-truth labels live in:
    # data/processed/labels/val or data/processed/labels/test
    labels_dir = DATA_ROOT / "labels" / split

    if not labels_dir.exists():
        print(f"ERROR: Labels dir not found: {labels_dir}")
        sys.exit(1)

    # ----------------------------
    # Metric counters
    # ----------------------------

    # Detection counters
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Classification counters
    total_matched = 0
    total_cls_correct = 0

    # Bookkeeping counters
    n_images = 0
    skipped_no_label = 0
    skipped_no_image = 0

    # ----------------------------
    # Evaluate each image
    # ----------------------------

    for rel_key, img_entry in images_data.items():
        # Get detections for this image from Step 3
        dets = img_entry.get("dets", [])

        # Try to reconstruct the image path from the relative key
        img_path = PROJECT_ROOT / rel_key

        if not img_path.exists():
            # Fallback: build path using the image file stem
            stem = Path(rel_key).stem
            img_path = DATA_ROOT / "images" / split / (stem + ".jpg")
            if not img_path.exists():
                img_path = DATA_ROOT / "images" / split / (stem + ".png")

        # Skip if image file cannot be found
        if not img_path.exists():
            skipped_no_image += 1
            continue

        n_images += 1

        # Open image to get width and height
        try:
            w, h = Image.open(img_path).size
        except Exception:
            skipped_no_image += 1
            continue

        # Matching ground-truth label file
        stem = img_path.stem
        label_path = labels_dir / (stem + ".txt")

        # Skip if label file is missing
        if not label_path.exists():
            skipped_no_label += 1
            continue

        # Load GT boxes for this image
        gt_boxes = load_gt_boxes(label_path, w, h)

        # If both are empty, nothing to score
        if not gt_boxes and not dets:
            continue

        # If no GT boxes but there are detections, all detections are false positives
        if not gt_boxes:
            total_fp += len(dets)
            continue

        # If GT boxes exist but no detections, all GT boxes are false negatives
        if not dets:
            total_fn += len(gt_boxes)
            continue

        # Extract prediction boxes only (ignore class for detection matching)
        pred_xyxy = [d["xyxy"] for d in dets]

        # Match predictions to GT boxes using IoU
        matches = match_predictions_to_gt(pred_xyxy, gt_boxes, args.iou)

        # Number of matched boxes = true positives for detection/localisation
        tp = len(matches)
        total_tp += tp

        # Predictions that did not match any GT = false positives
        total_fp += len(dets) - tp

        # GT boxes that did not get matched = false negatives
        total_fn += len(gt_boxes) - tp

        # Count how many matched boxes we have in total
        total_matched += tp

        # For each matched pair, check whether the predicted class is also correct
        for pi, gi in matches:
            pred_cls = dets[pi]["cls"]
            gt_cls = gt_boxes[gi][0]

            if pred_cls == gt_cls:
                total_cls_correct += 1

    # ----------------------------
    # Print warnings if needed
    # ----------------------------

    if skipped_no_image:
        print(f"Warning: Skipped {skipped_no_image} images (file not found).")

    if skipped_no_label:
        print(f"Warning: Skipped {skipped_no_label} images (no label file).")

    # ----------------------------
    # Detection metrics
    # ----------------------------

    # Total number of predictions
    total_preds = total_tp + total_fp

    # Total number of ground-truth boxes
    total_gt = total_tp + total_fn

    # Detection precision:
    # Of all predicted boxes, how many were correct matches?
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    # Detection recall:
    # Of all real GT boxes, how many did we find?
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # Detection F1:
    # Balanced score combining precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Classification accuracy on matched detections only
    # Of the predicted boxes that matched real boxes, how many had the right class?
    cls_accuracy = total_cls_correct / total_matched if total_matched > 0 else 0.0

    # End-to-end metrics: a prediction counts as correct only if (1) box matches GT (IoU >= 0.5)
    # and (2) the classifier predicted the right class. So TP = matched boxes with correct class.
    #   tp_e2e  = number of correct end-to-end predictions (matched + right class)
    #   fp_e2e  = predictions that were wrong (either no match or wrong class) = total_preds - tp_e2e
    #   fn_e2e  = ground-truth cells we missed or got wrong = total_gt - tp_e2e
    #   prec_e2e = tp_e2e / (tp_e2e + fp_e2e)  "Of what we predicted, how much was correct?"
    #   rec_e2e  = tp_e2e / (tp_e2e + fn_e2e)  "Of all GT cells, how much did we get right?"
    #   f1_e2e   = harmonic mean of precision and recall (standard F1 formula)
    tp_e2e = total_cls_correct
    fp_e2e = total_preds - tp_e2e
    fn_e2e = total_gt - tp_e2e
    prec_e2e = tp_e2e / (tp_e2e + fp_e2e) if (tp_e2e + fp_e2e) > 0 else 0.0
    rec_e2e = tp_e2e / (tp_e2e + fn_e2e) if (tp_e2e + fn_e2e) > 0 else 0.0
    f1_e2e = 2 * prec_e2e * rec_e2e / (prec_e2e + rec_e2e) if (prec_e2e + rec_e2e) > 0 else 0.0

    # ----------------------------
    # Print results
    # ----------------------------

    print()
    print("Step 4: Two-stage pipeline evaluation")
    print("=" * 50)
    print(f"Split: {split}")
    print(f"Predictions: {pred_path.name}")
    print(f"IoU threshold: {args.iou}")
    print(f"Images evaluated: {n_images}")
    print()

    # Detection/localisation metrics
    print("Detection / localisation (TP = IoU match only)")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print()

    # Full pipeline metrics
    print("End-to-end (TP = IoU match + correct class)")
    print(f"  TP: {tp_e2e}, FP: {fp_e2e}, FN: {fn_e2e}")
    print(f"  Precision: {prec_e2e:.4f}")
    print(f"  Recall:    {rec_e2e:.4f}")
    print(f"  F1:        {f1_e2e:.4f}")
    print()

    # Classification-only score on matched boxes
    print("Classification (on matched detections only)")
    print(f"  Matched: {total_matched}, Correct class: {total_cls_correct}")
    print(f"  Accuracy: {cls_accuracy:.4f}")
    print()

    # Remind user how to compare this against the YOLO end-to-end script
    print("Compare with end-to-end YOLO: python3 scripts/class_imbalance/evaluate_conditions.py --split", split)

if __name__ == "__main__":
    main()
