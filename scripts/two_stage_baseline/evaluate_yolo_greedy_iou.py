"""
Evaluate YOLO with the same greedy IoU evaluator used by two-stage Step 4.

What this script does:
1) Run YOLO detector on val/test images.
2) Save detections in Step 3 JSON format (xyxy + cls + det_conf).
3) Call step4_evaluate_two_stage.py on that JSON so matching/TP/FP/FN are identical.

Run from project root:
  python3 scripts/two_stage_baseline/evaluate_yolo_greedy_iou.py --split test
  python3 scripts/two_stage_baseline/evaluate_yolo_greedy_iou.py --split val --conf 0.25
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "yolo_greedy_eval"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
STEP4 = PROJECT_ROOT / "scripts" / "two_stage_baseline" / "step4_evaluate_two_stage.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO -> Step3 JSON -> Step4 greedy IoU evaluation")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--yolo_weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="Path to YOLO best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold used by Step 4 matcher")
    parser.add_argument("--images_dir", type=Path, default=None, help="Optional override for images directory")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Where to write predictions JSON")
    parser.add_argument("--suffix", type=str, default="yolo_greedy", help="Output filename suffix")
    parser.add_argument(
        "--skip_step4",
        action="store_true",
        help="Only export predictions JSON, do not call Step 4 evaluator",
    )
    args = parser.parse_args()

    if not args.yolo_weights.exists():
        print(f"ERROR: YOLO weights not found: {args.yolo_weights}")
        sys.exit(1)
    if not STEP4.exists():
        print(f"ERROR: Step 4 script not found: {STEP4}")
        sys.exit(1)

    if args.images_dir is not None:
        img_dir = args.images_dir
    else:
        img_dir = DATA_ROOT / args.split

    if not img_dir.exists():
        print(f"ERROR: Image dir not found: {img_dir}")
        sys.exit(1)

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not image_paths:
        print(f"ERROR: No images found in: {img_dir}")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(args.yolo_weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {"split": args.split, "images": {}}

    for idx, img_path in enumerate(image_paths):
        img = np.array(Image.open(img_path).convert("RGB"))
        result = model.predict(source=img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        dets = []
        if result.boxes is not None:
            xyxy_tensor = result.boxes.xyxy
            conf_tensor = result.boxes.conf
            cls_tensor = result.boxes.cls
            n_boxes = xyxy_tensor.shape[0]

            for j in range(n_boxes):
                dets.append(
                    {
                        "xyxy": xyxy_tensor[j].cpu().numpy().tolist(),
                        "det_conf": float(conf_tensor[j].cpu().item()),
                        "cls": int(cls_tensor[j].cpu().item()),
                    }
                )

        try:
            rel_path = str(img_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = img_path.name
        predictions["images"][rel_path] = {"dets": dets}

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"  {idx + 1}/{len(image_paths)} images")

    predictions["meta"] = {
        "mode": "yolo_only_greedy_step4",
        "yolo_weights": str(args.yolo_weights),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "iou_for_step4": args.iou,
        "split": args.split,
    }

    out_name = f"predictions_{args.split}_{args.suffix}.json"
    out_file = args.output_dir / out_name
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved YOLO predictions JSON: {out_file}")

    if args.skip_step4:
        print("Skipping Step 4 as requested (--skip_step4).")
        return

    print("\nRunning Step 4 with identical matching rules...")
    cmd = [
        sys.executable,
        str(STEP4),
        "--predictions",
        str(out_file),
        "--iou",
        str(args.iou),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
