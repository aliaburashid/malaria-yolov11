"""
Oracle crop evaluation: classifier accuracy on ground-truth crops only.

This script does NOT run object detection. It crops each cell using the
label-file boxes (perfect localisation) and runs the Stage-2 classifier.
Use it to compare against Step 4's "Classification (on matched detections only)",
which uses crops from YOLO boxes after IoU matching.

Fairness note (for write-up):
  Oracle evaluates every GT cell on the split. Step 4 classification accuracy is
  only on IoU-matched YOLO detections—a subset when the detector misses cells.
  The comparison is therefore not strictly apples-to-apples; oracle is still a
  useful upper bound on classifier performance under perfect localisation, and
  contrasting it with matched-detection accuracy indicates whether imperfect
  detector crops contribute to the pipeline gap.

Run from project root:
  python3 scripts/two_stage_baseline/step_oracle_crop_eval.py --split val
  python3 scripts/two_stage_baseline/step_oracle_crop_eval.py --split test
  python3 scripts/two_stage_baseline/step_oracle_crop_eval.py --split test --classifier_weights runs/classifier_27k_finetuned/best.pt
  python3 scripts/two_stage_baseline/step_oracle_crop_eval.py --split val --suffix finetuned

Optional:
  --crop_pad 0.1   (default: same as step3_two_stage_inference.CROP_PAD; use 0 for no padding)
  --suffix NAME    optional tag for output JSON (avoids overwriting when comparing checkpoints)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
IMAGES_ROOT = DATA_ROOT / "images"
LABELS_ROOT = DATA_ROOT / "labels"

DEFAULT_CLASSIFIER = PROJECT_ROOT / "runs" / "classifier_27k" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "two_stage_baseline"

_BASELINE_DIR = Path(__file__).resolve().parent


def _load_sibling_module(stem: str):
    """Load step3 / step4 as modules without package install (same folder)."""
    path = _BASELINE_DIR / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing sibling script: {path}")
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_step3 = _load_sibling_module("step3_two_stage_inference")
_step4 = _load_sibling_module("step4_evaluate_two_stage")

load_classifier = _step3.load_classifier
crop_box = _step3.crop_box
load_gt_boxes = _step4.load_gt_boxes
CLASS_NAMES = _step4.CLASS_NAMES


def collect_image_paths(split: str) -> list[Path]:
    img_dir = IMAGES_ROOT / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Oracle crop eval: classifier on GT boxes only (no YOLO)."
    )
    parser.add_argument("--split", choices=["val", "test"], required=True, help="val or test")
    parser.add_argument(
        "--classifier_weights",
        type=Path,
        default=DEFAULT_CLASSIFIER,
        help="Classifier checkpoint (default: runs/classifier_27k/best.pt)",
    )
    parser.add_argument(
        "--crop_pad",
        type=float,
        default=float(_step3.CROP_PAD),
        help="Fractional padding around GT box before crop (default: same as Step 3, "
        f"currently {_step3.CROP_PAD}; use 0 for tight GT box only)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write oracle_crop_eval_{split}[_{suffix}].json",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for output filename, e.g. finetuned -> oracle_crop_eval_val_finetuned.json",
    )
    args = parser.parse_args()
    args.classifier_weights = Path(args.classifier_weights)

    if not args.classifier_weights.exists():
        print(f"ERROR: Classifier weights not found: {args.classifier_weights}")
        sys.exit(1)

    labels_dir = LABELS_ROOT / args.split
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, transform, _img_size = load_classifier(args.classifier_weights, device)

    image_paths = collect_image_paths(args.split)
    if not image_paths:
        print(f"No images under {IMAGES_ROOT / args.split}")
        sys.exit(1)

    # Counters: overall and per class (0 = parasitized, 1 = uninfected)
    total = 0
    correct = 0
    per_n = [0, 0]
    per_correct = [0, 0]

    skipped_no_label = 0
    skipped_empty_crop = 0
    n_images_used = 0

    n_paths = len(image_paths)
    for idx, img_path in enumerate(image_paths):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped_no_label += 1
            continue

        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_rgb.shape[0], img_rgb.shape[1]

        gt_boxes = load_gt_boxes(label_path, w, h)
        if not gt_boxes:
            continue

        n_images_used += 1

        for cls_id, xyxy in gt_boxes:
            if cls_id not in (0, 1):
                continue

            crop = crop_box(img_rgb, xyxy, pad=args.crop_pad)
            if crop.size == 0:
                skipped_empty_crop += 1
                continue

            pil_crop = Image.fromarray(crop)
            tensor = transform(pil_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                out = classifier(tensor)
                pred = int(out.argmax(dim=1).item())

            total += 1
            per_n[cls_id] += 1
            if pred == cls_id:
                correct += 1
                per_correct[cls_id] += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == n_paths:
            print(f"  {idx + 1}/{n_paths} images", flush=True)

    overall_acc = correct / total if total > 0 else 0.0

    def _safe_acc(c: int, n: int) -> float:
        return c / n if n > 0 else 0.0

    parasitized_acc = _safe_acc(per_correct[0], per_n[0])
    uninfected_acc = _safe_acc(per_correct[1], per_n[1])

    result = {
        "experiment": "oracle_crop_classification",
        "split": args.split,
        "classifier_weights": str(args.classifier_weights),
        "crop_pad": args.crop_pad,
        "n_images_with_at_least_one_gt": n_images_used,
        "n_gt_crops_evaluated": total,
        "n_correct": correct,
        "overall_accuracy": round(overall_acc, 6),
        "parasitized": {
            "n": per_n[0],
            "correct": per_correct[0],
            "accuracy": round(parasitized_acc, 6),
        },
        "uninfected": {
            "n": per_n[1],
            "correct": per_correct[1],
            "accuracy": round(uninfected_acc, 6),
        },
        "skipped_no_label_file": skipped_no_label,
        "skipped_empty_crop_after_pad_clip": skipped_empty_crop,
        "class_names": CLASS_NAMES,
        "output_suffix": args.suffix or None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix_part = f"_{args.suffix}" if args.suffix else ""
    out_path = args.output_dir / f"oracle_crop_eval_{args.split}{suffix_part}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("Oracle crop evaluation (classifier on GT crops only; no YOLO)")
    print("=" * 55)
    print(f"  Split:                    {args.split}")
    print(f"  Classifier weights:       {args.classifier_weights}")
    print(f"  Crop pad:                 {args.crop_pad}")
    print(f"  GT crops evaluated:       {total}")
    print(f"  Overall accuracy:         {overall_acc:.4f}")
    print(f"  Parasitized ({CLASS_NAMES[0]}): n={per_n[0]}, acc={parasitized_acc:.4f}")
    print(f"  Uninfected ({CLASS_NAMES[1]}): n={per_n[1]}, acc={uninfected_acc:.4f}")
    if skipped_no_label:
        print(f"  Warning: skipped {skipped_no_label} images (no label .txt)")
    if skipped_empty_crop:
        print(f"  Warning: skipped {skipped_empty_crop} GT boxes (empty crop after clip)")
    print()
    print(f"  Saved: {out_path}")
    print()
    print("Compare with Step 4 line: Classification (on matched detections only)")
    print(f"  python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split {args.split}")


if __name__ == "__main__":
    main()
