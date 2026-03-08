"""
Step 3: Two-stage inference — YOLO (detector only) → crop → CNN (classifier).

Loads best YOLO and the Stage-2 classifier. For each val/test image: run YOLO to get
boxes (ignore YOLO class); crop each box; run classifier on each crop; output (box, class from CNN).
Saves predictions to runs/two_stage_baseline/predictions_{split}.json for Step 4.

Run from project root:
  python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val
  python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split test
Sources: Ultralytics YOLO predict (https://docs.ultralytics.com/); Step 2 checkpoint format.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# Project root (this file lives in scripts/two_stage_baseline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "images"
CLASSIFIER_PATH = PROJECT_ROOT / "runs" / "classifier_27k" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "two_stage_baseline"
# Default YOLO weights: best condition (D). Override with --yolo_weights.
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
CONF_THRESH = 0.25
# Padding around crop (fraction of box size) before resizing to classifier input
CROP_PAD = 0.1


def load_classifier(ckpt_path: Path, device: torch.device):
    """Load Stage-2 classifier from Step 2 checkpoint. Rebuild model from saved arch, load state_dict (pretrained choice irrelevant at inference)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "resnet18")
    img_size = ckpt.get("img_size", 224)
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch: {arch}")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return model, transform, img_size


def crop_box(img: np.ndarray, xyxy: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Crop image to box with optional padding. xyxy: [x1,y1,x2,y2] in pixel coords.
    Always clips coords to [0,w]/[0,h]. Returns empty array if box is invalid (x2 <= x1 or y2 <= y1).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(float)
    bw, bh = x2 - x1, y2 - y1
    if pad > 0:
        x1 = x1 - bw * pad
        y1 = y1 - bh * pad
        x2 = x2 + bw * pad
        y2 = y2 + bh * pad
    # Always clip to image bounds (YOLO can output coords slightly outside)
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]  # invalid/tiny box: skip in caller via crop.size == 0
    return img[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="Two-stage inference: YOLO detect → crop → CNN classify")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="val or test set")
    parser.add_argument("--yolo_weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="Path to YOLO best.pt")
    parser.add_argument("--classifier_weights", type=Path, default=CLASSIFIER_PATH, help="Path to classifier best.pt")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size (match train/eval for consistency)")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Where to save predictions JSON")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for output filename, e.g. 'finetuned' -> predictions_val_finetuned.json (keeps baseline and finetuned results separate)")
    args = parser.parse_args()
    args.yolo_weights = Path(args.yolo_weights)
    args.classifier_weights = Path(args.classifier_weights)

    if not args.yolo_weights.exists():
        print(f"ERROR: YOLO weights not found: {args.yolo_weights}")
        sys.exit(1)
    if not args.classifier_weights.exists():
        print(f"ERROR: Classifier weights not found: {args.classifier_weights}. Run step2 first.")
        sys.exit(1)

    img_dir = DATA_ROOT / args.split
    if not img_dir.exists():
        print(f"ERROR: Image dir not found: {img_dir}")
        sys.exit(1)
    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not image_paths:
        print(f"No images in {img_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ultralytics import YOLO

    yolo = YOLO(str(args.yolo_weights))
    classifier, transform, img_size = load_classifier(args.classifier_weights, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {"split": args.split, "images": {}}

    for i, img_path in enumerate(image_paths):
        img = np.array(Image.open(img_path).convert("RGB"))
        results = yolo.predict(source=img, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]
        dets = []  # Per-detection list of dicts: avoids length desync; Step 4 simpler.
        # Stable API: use Boxes tensors (Nx4, N) instead of iterating result.boxes. Ultralytics docs.
        if result.boxes is not None:
            xyxy_tensor = result.boxes.xyxy   # Nx4
            conf_tensor = result.boxes.conf   # N
            n_boxes = xyxy_tensor.shape[0]
            for j in range(n_boxes):
                xyxy = xyxy_tensor[j].cpu().numpy()
                det_conf = float(conf_tensor[j].cpu().item())
                crop = crop_box(img, xyxy, pad=CROP_PAD)
                if crop.size == 0:
                    continue
                pil_crop = Image.fromarray(crop)
                tensor = transform(pil_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = classifier(tensor)
                    probs = torch.softmax(out, dim=1)
                    cls = int(out.argmax(dim=1).item())
                    cls_conf = float(probs[0, cls].item())
                dets.append({
                    "xyxy": xyxy.tolist(),
                    "det_conf": det_conf,
                    "cls": cls,
                    "cls_conf": cls_conf,
                })
        try:
            rel_path = str(img_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = img_path.name
        predictions["images"][rel_path] = {"dets": dets}
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  {i + 1}/{len(image_paths)} images")

    predictions["meta"] = {
        "yolo_weights": str(args.yolo_weights),
        "classifier_weights": str(args.classifier_weights),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "crop_pad": CROP_PAD,
        "suffix": args.suffix or None,
    }
    name = f"predictions_{args.split}{'_' + args.suffix if args.suffix else ''}.json"
    out_file = args.output_dir / name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved: {out_file}")
    print("Next: python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py")


if __name__ == "__main__":
    main()
