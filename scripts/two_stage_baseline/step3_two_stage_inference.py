"""
Step 3: Two-stage inference — YOLO (detector only) -> crop -> CNN (classifier).

What this script does:
- Loads the best YOLO detector and the Stage-2 classifier
- Runs YOLO on each val/test image to get bounding boxes
- Ignores YOLO's class label and uses YOLO only for localisation
- Crops each detected box from the original image
- Runs the CNN classifier on each crop
- Saves all detections + classifier outputs to JSON for Step 4

Run from project root:
  python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val
  python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split test

Main sources used in this script:
- Ultralytics YOLO predict API: https://docs.ultralytics.com/
- torchvision ResNet18 + transforms: https://pytorch.org/vision/stable/index.html
- Step 2 / Step 2b checkpoint format from the project
"""

# argparse lets us add terminal options like --split val or --suffix finetuned
import argparse

# json is used to save predictions in a structured file for Step 4 evaluation
import json

# sys is used so we can stop the script early if files/folders are missing
import sys

# Path helps build safe file/folder paths across Mac/Linux/Windows
from pathlib import Path

# NumPy is used for image arrays and box coordinate handling
import numpy as np

# PyTorch is used for loading the classifier checkpoint and running inference
import torch

# Source: torchvision models/transforms documentation
# models = used to rebuild ResNet18
# transforms = used to resize/normalize crops before classification
from torchvision import models, transforms

# PIL is used to open images and convert crops back into image objects for transforms
from PIL import Image

# Project root:
# This file lives inside scripts/two_stage_baseline/, so go up 3 levels to repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# DATA_ROOT points to val/test images used for two-stage inference.
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "images"

# Default classifier checkpoint from Step 2 (27k-only classifier).
CLASSIFIER_PATH = PROJECT_ROOT / "runs" / "classifier_27k" / "best.pt"

# Where Step 3 will save prediction JSON files for Step 4.
OUTPUT_DIR = PROJECT_ROOT / "runs" / "two_stage_baseline"

# Default YOLO weights:
# Best end-to-end model from the experiments (Condition D = oversampled + weighted).
# Can be overridden with --yolo_weights if needed.
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"

# Default YOLO confidence threshold.
# Lower threshold = more detections (higher recall, sometimes more false positives).
CONF_THRESH = 0.25

# Padding added around each YOLO box before cropping.
# Why: a slightly larger crop often helps the classifier see the full cell context.
CROP_PAD = 0.1


def load_classifier(ckpt_path: Path, device: torch.device):
    """
    Load the Stage-2 classifier from a Step 2 / Step 2b checkpoint.

    What it does:
    - reads the saved checkpoint file
    - rebuilds the ResNet18 architecture
    - loads trained weights
    - creates the same image transform used during classifier training

    Why:
    - Step 3 must classify crops in the same way the classifier was trained.
    """

    # Source: torch.load documentation
    # Load checkpoint from disk onto CPU or GPU depending on selected device.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Read metadata saved in the checkpoint.
    # arch tells us which model architecture was used.
    # img_size tells us what size the classifier expects.
    arch = ckpt.get("arch", "resnet18")
    img_size = ckpt.get("img_size", 224)

    # This script currently only supports ResNet18 checkpoints.
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch: {arch}")

    # Source: torchvision ResNet18 documentation
    # Rebuild the classifier architecture.
    # weights=None because we are not using ImageNet weights at inference;
    # we are loading our own trained weights from the checkpoint right after.
    model = models.resnet18(weights=None)

    # Replace final fully connected layer so it outputs 2 classes:
    # 0 = parasitized, 1 = uninfected
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # Load the trained model weights from Step 2 / Step 2b.
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Move model to selected device (CPU or GPU).
    model = model.to(device)

    # Set evaluation mode so batch norm / dropout behave correctly for inference.
    model.eval()

    # Source: same normalization used in Step 2 classifier training
    # These are ImageNet normalization values because ResNet18 was trained that way.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Source: torchvision transforms documentation
    # Every crop is resized to classifier input size, converted to tensor, then normalized.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Return:
    # - the loaded classifier
    # - the preprocessing transform for crops
    # - the image size recorded in the checkpoint
    return model, transform, img_size


def crop_box(img: np.ndarray, xyxy: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """
    Crop one detection box from the full image.

    Input:
    - img: full RGB image as NumPy array
    - xyxy: box coordinates [x1, y1, x2, y2] in pixel space
    - pad: optional fraction of extra padding around the box

    Why:
    - YOLO gives a box around a cell
    - the classifier needs the actual cropped image region inside that box

    Important:
    - coordinates are clipped to image boundaries
    - invalid boxes return an empty crop
    """

    # Get image height and width.
    h, w = img.shape[:2]

    # Convert box coordinates to float for safe padding math.
    x1, y1, x2, y2 = xyxy.astype(float)

    # Compute box width and height.
    bw, bh = x2 - x1, y2 - y1

    # Add padding around the box if requested.
    if pad > 0:
        x1 = x1 - bw * pad
        y1 = y1 - bh * pad
        x2 = x2 + bw * pad
        y2 = y2 + bh * pad

    # Clip coordinates to stay within the image.
    # This avoids slicing outside array bounds.
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    # Convert to ints for NumPy slicing.
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # If the box becomes invalid after clipping, return an empty crop.
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]

    # Return the cropped region [rows, cols].
    return img[y1:y2, x1:x2]



def main():
    # Source: argparse documentation
    # Create terminal arguments so the same script can run on val or test,
    # and can use different classifier or YOLO weights if needed.
    parser = argparse.ArgumentParser(description="Two-stage inference: YOLO detect -> crop -> CNN classify")

    # Choose which data split to run on.
    parser.add_argument("--split", choices=["val", "test"], default="val", help="val or test set")

    # Optional path to YOLO weights; by default uses best Condition D weights.
    parser.add_argument("--yolo_weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="Path to YOLO best.pt")

    # Optional path to classifier weights; by default uses Step 2 baseline classifier.
    parser.add_argument("--classifier_weights", type=Path, default=CLASSIFIER_PATH, help="Path to classifier best.pt")

    # Optional confidence threshold for YOLO detections.
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="YOLO confidence threshold")

    # YOLO input image size, kept consistent with train/eval settings.
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size (match train/eval for consistency)")

    # Folder where the prediction JSON will be saved.
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="Where to save predictions JSON")

    # Optional suffix to avoid overwriting baseline results, e.g. finetuned.
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix for output filename, e.g. 'finetuned' -> predictions_val_finetuned.json (keeps baseline and finetuned results separate)")
    # Override image directory (e.g. for robustness: corrupted test images).
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Override image directory (default: data/processed/images/<split>). Used e.g. for robustness on corrupted sets.")

    # Parse terminal arguments.
    args = parser.parse_args()
    # Convert paths into Path objects in case they came in as strings.
    args.yolo_weights = Path(args.yolo_weights)
    args.classifier_weights = Path(args.classifier_weights)

   # ----------------------------
    # Safety checks
    # ----------------------------

    # Stop if YOLO weights do not exist.
    if not args.yolo_weights.exists():
        print(f"ERROR: YOLO weights not found: {args.yolo_weights}")
        sys.exit(1)

    # Stop if classifier checkpoint does not exist.
    if not args.classifier_weights.exists():
        print(f"ERROR: Classifier weights not found: {args.classifier_weights}. Run step2 first.")
        sys.exit(1)

    # Build path to the selected split folder, or use override (e.g. robustness corrupted set).
    if args.images_dir is not None:
        img_dir = Path(args.images_dir)
    else:
        img_dir = DATA_ROOT / args.split

    # Stop if that split folder is missing.
    if not img_dir.exists():
        print(f"ERROR: Image dir not found: {img_dir}")
        sys.exit(1)

    # Collect all image files in the split.
    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    # Stop if no images are found.
    if not image_paths:
        print(f"No images in {img_dir}")
        sys.exit(1)

    # ----------------------------
    # Device + model loading
    # ----------------------------

    # Source: PyTorch device selection
    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Source: Ultralytics package
    # Import YOLO only when needed inside main.
    from ultralytics import YOLO

    # Load YOLO detector from weights file.
    yolo = YOLO(str(args.yolo_weights))

    # Load classifier and crop transform from checkpoint.
    classifier, transform, img_size = load_classifier(args.classifier_weights, device)

    # Create output folder if it doesn't exist.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # This dictionary will hold all predictions for the whole split.
    predictions = {"split": args.split, "images": {}}

    # ----------------------------
    # Main inference loop
    # ----------------------------

    # Loop over each image in val/test.
    for i, img_path in enumerate(image_paths):

        # Load image as RGB NumPy array.
        img = np.array(Image.open(img_path).convert("RGB"))

        # Source: Ultralytics YOLO predict API
        # Run YOLO detection on the image.
        # Important: we use YOLO only for boxes here, not for final class labels.
        results = yolo.predict(source=img, conf=args.conf, imgsz=args.imgsz, verbose=False)

        # There is only one input image here, so use the first result.
        result = results[0]

        # Store detections for this image here.
        # One dict per detection avoids syncing separate lists later.
        dets = []

        # If YOLO found any boxes, process them one by one.
        if result.boxes is not None:
            # Source: Ultralytics Boxes object
            # xyxy_tensor has shape [N, 4], conf_tensor has shape [N]
            xyxy_tensor = result.boxes.xyxy
            conf_tensor = result.boxes.conf

            # Number of detections in this image.
            n_boxes = xyxy_tensor.shape[0]

            # Loop through each YOLO box.
            for j in range(n_boxes):
                # Convert one box to NumPy on CPU.
                xyxy = xyxy_tensor[j].cpu().numpy()

                # YOLO detector confidence for this box.
                det_conf = float(conf_tensor[j].cpu().item())

                # Crop this box from the full image.
                crop = crop_box(img, xyxy, pad=CROP_PAD)

                # Skip invalid or empty crops.
                if crop.size == 0:
                    continue

                # Convert crop back to PIL format so torchvision transforms can be applied.
                pil_crop = Image.fromarray(crop)

                # Apply classifier preprocessing and add batch dimension with unsqueeze(0).
                tensor = transform(pil_crop).unsqueeze(0).to(device)

                # Source: PyTorch inference best practice
                # No gradients needed because this is inference only.
                with torch.no_grad():
                    # Run classifier on the crop.
                    out = classifier(tensor)

                    # Convert logits to probabilities.
                    probs = torch.softmax(out, dim=1)

                    # Predicted class index: 0 or 1.
                    cls = int(out.argmax(dim=1).item())

                    # Confidence for the predicted class.
                    cls_conf = float(probs[0, cls].item())

                # Save one detection entry:
                # - box coordinates from YOLO
                # - detector confidence
                # - classifier predicted class
                # - classifier confidence
                dets.append({
                    "xyxy": xyxy.tolist(),
                    "det_conf": det_conf,
                    "cls": cls,
                    "cls_conf": cls_conf,
                })

        # Save image path relative to repo root when possible.
        # This makes prediction files portable inside the project.
        try:
            rel_path = str(img_path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel_path = img_path.name

        # Save detections for this image into the JSON structure.
        predictions["images"][rel_path] = {"dets": dets}

        # Print progress every 50 images and at the final image.
        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  {i + 1}/{len(image_paths)} images")

    # ----------------------------
    # Save metadata + predictions
    # ----------------------------

    # Save run settings into the JSON for reproducibility.
    predictions["meta"] = {
        "yolo_weights": str(args.yolo_weights),
        "classifier_weights": str(args.classifier_weights),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "crop_pad": CROP_PAD,
        "suffix": args.suffix or None,
    }

    # Build output filename.
    # Examples:
    # predictions_val.json
    # predictions_test.json
    # predictions_val_finetuned.json
    name = f"predictions_{args.split}{'_' + args.suffix if args.suffix else ''}.json"
    out_file = args.output_dir / name

    # Make sure output directory exists before writing.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write the full predictions dictionary to JSON for Step 4.
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)

    # Print where predictions were saved and the next step.
    print(f"Saved: {out_file}")
    print("Next: python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py")


if __name__ == "__main__":
    main()
