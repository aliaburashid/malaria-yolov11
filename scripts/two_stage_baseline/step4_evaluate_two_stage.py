"""
Step 4: Evaluate two-stage pipeline against ground truth.

Loads predictions from Step 3 (predictions_{split}.json) and GT labels (YOLO format).
Matches predicted boxes to GT by IoU >= 0.5; for matched pairs, checks if Stage-2 class is correct.
Reports detection metrics (Precision, Recall, F1) and classification accuracy (on matched detections).

Run from project root:
  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split val
  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split test
  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --predictions runs/two_stage_baseline/predictions_val.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Project root (this file lives in scripts/two_stage_baseline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "runs" / "two_stage_baseline"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
IOU_THRESH = 0.5
CLASS_NAMES = ["parasitized", "uninfected"]


def box_iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> float:
    """Intersection over union of two boxes. Each box: [x1, y1, x2, y2]."""
    x1 = max(a_xyxy[0], b_xyxy[0])
    y1 = max(a_xyxy[1], b_xyxy[1])
    x2 = min(a_xyxy[2], b_xyxy[2])
    y2 = min(a_xyxy[3], b_xyxy[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    # Guard against invalid/negative-area boxes
    area_a = max(0.0, a_xyxy[2] - a_xyxy[0]) * max(0.0, a_xyxy[3] - a_xyxy[1])
    area_b = max(0.0, b_xyxy[2] - b_xyxy[0]) * max(0.0, b_xyxy[3] - b_xyxy[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """Load YOLO-format labels (class_id x_center y_center width height, normalized). Return list of (class_id, xyxy pixel).
    Clips coords to [0,w]/[0,h] and skips invalid boxes (x2<=x1 or y2<=y1) for stable IoU."""
    if not label_path.exists():
        return []
    out = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float64)))
    return out


def match_predictions_to_gt(
    pred_xyxy_list: list, gt_boxes: list[tuple[int, np.ndarray]], iou_thresh: float
) -> list[tuple[int, int]]:
    """Greedy one-to-one match: (pred_idx, gt_idx) with IoU >= iou_thresh. Sorted by IoU descending."""
    if not pred_xyxy_list or not gt_boxes:
        return []
    pred_arr = np.array(pred_xyxy_list, dtype=np.float64)
    gt_xyxy = np.array([g[1] for g in gt_boxes], dtype=np.float64)
    # IoU matrix: pred x gt
    n_p, n_g = len(pred_arr), len(gt_xyxy)
    ious = np.zeros((n_p, n_g))
    for i in range(n_p):
        for j in range(n_g):
            ious[i, j] = box_iou(pred_arr[i], gt_xyxy[j])
    # Greedy: collect (iou, pred_idx, gt_idx), sort by iou desc, then assign
    candidates = []
    for i in range(n_p):
        for j in range(n_g):
            if ious[i, j] >= iou_thresh:
                candidates.append((ious[i, j], i, j))
    candidates.sort(key=lambda x: -x[0])
    matched_p, matched_g = set(), set()
    result = []
    for _iou, pi, gi in candidates:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        result.append((pi, gi))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-stage pipeline vs GT")
    parser.add_argument("--split", choices=["val", "test"], default=None, help="Use predictions_{split}.json (or with --suffix: predictions_{split}_{suffix}.json)")
    parser.add_argument("--predictions", type=Path, default=None, help="Path to predictions JSON (overrides --split/--suffix/--predictions_dir)")
    parser.add_argument("--predictions_dir", type=Path, default=PREDICTIONS_DIR, help="Directory containing predictions JSON (default: runs/two_stage_baseline)")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix to match Step 3 output, e.g. 'finetuned' -> predictions_val_finetuned.json")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="IoU threshold for match (default 0.5)")
    args = parser.parse_args()

    if args.predictions is not None:
        pred_path = Path(args.predictions)
    elif args.split is not None:
        name = f"predictions_{args.split}{'_' + args.suffix if args.suffix else ''}.json"
        pred_path = Path(args.predictions_dir) / name
    else:
        print("ERROR: Provide --split val|test or --predictions <path>")
        sys.exit(1)

    if not pred_path.exists():
        print(f"ERROR: Predictions file not found: {pred_path}")
        print("Run Step 3 first: python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val")
        sys.exit(1)

    with open(pred_path) as f:
        data = json.load(f)
    split = data.get("split", args.split or "val")
    images_data = data.get("images", {})
    if not images_data:
        print("ERROR: No images in predictions JSON")
        sys.exit(1)

    labels_dir = DATA_ROOT / "labels" / split
    if not labels_dir.exists():
        print(f"ERROR: Labels dir not found: {labels_dir}")
        sys.exit(1)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_matched = 0
    total_cls_correct = 0
    n_images = 0
    skipped_no_label = 0
    skipped_no_image = 0

    for rel_key, img_entry in images_data.items():
        dets = img_entry.get("dets", [])
        # Resolve image path to get dimensions and stem for labels
        img_path = PROJECT_ROOT / rel_key
        if not img_path.exists():
            # Try stem only (e.g. key might be just filename)
            stem = Path(rel_key).stem
            img_path = DATA_ROOT / "images" / split / (stem + ".jpg")
            if not img_path.exists():
                img_path = DATA_ROOT / "images" / split / (stem + ".png")
        if not img_path.exists():
            skipped_no_image += 1
            continue
        n_images += 1
        try:
            w, h = Image.open(img_path).size
        except Exception:
            skipped_no_image += 1
            continue
        stem = img_path.stem
        label_path = labels_dir / (stem + ".txt")
        if not label_path.exists():
            skipped_no_label += 1
            continue
        gt_boxes = load_gt_boxes(label_path, w, h)
        if not gt_boxes and not dets:
            continue
        if not gt_boxes:
            total_fp += len(dets)
            continue
        if not dets:
            total_fn += len(gt_boxes)
            continue

        pred_xyxy = [d["xyxy"] for d in dets]
        matches = match_predictions_to_gt(pred_xyxy, gt_boxes, args.iou)
        tp = len(matches)
        total_tp += tp
        total_fp += len(dets) - tp
        total_fn += len(gt_boxes) - tp
        total_matched += tp
        for pi, gi in matches:
            pred_cls = dets[pi]["cls"]
            gt_cls = gt_boxes[gi][0]
            if pred_cls == gt_cls:
                total_cls_correct += 1

    if skipped_no_image:
        print(f"Warning: Skipped {skipped_no_image} images (file not found).")
    if skipped_no_label:
        print(f"Warning: Skipped {skipped_no_label} images (no label file).")

    # Detection (localisation only: TP = IoU match, class ignored)
    total_preds = total_tp + total_fp
    total_gt = total_tp + total_fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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

    print()
    print("Step 4: Two-stage pipeline evaluation")
    print("=" * 50)
    print(f"Split: {split}")
    print(f"Predictions: {pred_path.name}")
    print(f"IoU threshold: {args.iou}")
    print(f"Images evaluated: {n_images}")
    print()
    print("Detection / localisation (TP = IoU match only)")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print()
    print("End-to-end (TP = IoU match + correct class)")
    print(f"  TP: {tp_e2e}, FP: {fp_e2e}, FN: {fn_e2e}")
    print(f"  Precision: {prec_e2e:.4f}")
    print(f"  Recall:    {rec_e2e:.4f}")
    print(f"  F1:        {f1_e2e:.4f}")
    print()
    print("Classification (on matched detections only)")
    print(f"  Matched: {total_matched}, Correct class: {total_cls_correct}")
    print(f"  Accuracy: {cls_accuracy:.4f}")
    print()
    print("Compare with end-to-end YOLO: python3 scripts/evaluate_conditions.py --split", split)


if __name__ == "__main__":
    main()
