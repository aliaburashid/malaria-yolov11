"""
Step 2b: Fine-tune the Stage-2 classifier on thin-smear GT crops.

Goal:
- Take the classifier trained in Step 2 on the NIH 27k cell dataset
- Fine-tune it on crops cut from the thin-smear TRAIN images
- Save a NEW classifier that is better matched to the real pipeline crops

Why this exists:
- The 27k NIH cells are clean, pre-cropped single-cell images
- But in the real two-stage pipeline, the classifier sees YOLO crops from full smear images
- Those crops may look different (more background, slightly off-center, different lighting)
- Fine-tuning helps the classifier adapt to the real crop style

Important:
- This does NOT overwrite the original classifier
- Original model stays at:
    runs/classifier_27k/best.pt
- Fine-tuned model is saved to:
    runs/classifier_27k_finetuned/best.pt

Run:
  python3 scripts/two_stage_baseline/step2b_finetune_classifier_thinsmear.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# PROJECT_ROOT = main repo folder
# This script lives in scripts/two_stage_baseline/, so go up 3 folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# DATA_ROOT = processed YOLO-style dataset folder
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

# These are the THIN-SMEAR TRAIN images and labels only
# We use TRAIN ONLY to avoid leaking val/test information into the classifier
TRAIN_IMAGES = DATA_ROOT / "images" / "train"
TRAIN_LABELS = DATA_ROOT / "labels" / "train"

# The classifier trained in Step 2 on the 27k NIH cell dataset
CLASSIFIER_27K = PROJECT_ROOT / "runs" / "classifier_27k" / "best.pt"

# New folder where the fine-tuned classifier will be saved
OUTPUT_DIR = PROJECT_ROOT / "runs" / "classifier_27k_finetuned"

# ----------------------------
# Settings / Hyperparameters
# ----------------------------
# Fixed seed for reproducibility
SEED = 42
# Same image size as Step 2 and Step 3
IMG_SIZE = 224
# Training settings
BATCH_SIZE = 32
EPOCHS = 10
# Small learning rate because we are fine-tuning an already-trained model,
# not training from scratch
LR = 5e-5
# Hold out 10% of the thin-smear GT crops for validation
VAL_RATIO = 0.1
# Add a little padding around GT crop boundaries
# Same idea as Step 3, so the fine-tune data resembles the real pipeline crops
CROP_PAD = 0.1


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """
    Load YOLO-format labels and convert them to pixel boxes.

    Input format of each label line:
      class_id x_center y_center width height
    where all coordinates are normalized (0 to 1)

    Output:
    - list of (class_id, xyxy_box_in_pixels)

    Why:
    - We need real crop coordinates to cut cells out of the thin-smear train images
    """
    # If label file does not exist, return empty list
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

            # Convert normalized xywh to pixel xyxy
            # Source: x1 = (x_center - width/2) * img_w; Ultralytics label format (normalized 0–1).
            # https://docs.ultralytics.com/datasets/detect/
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h

            # Clip coordinates so they stay inside image bounds
            # Source: standard practice; same as step4_evaluate_two_stage.load_gt_boxes and convert_to_yolo.
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            out.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float64)))

    return out




def build_crop_list():
    """
    Build a list of all GT crops from the THIN-SMEAR TRAIN set.

    Output format:
      (img_path, x1, y1, x2, y2, class_id)

    Why:
    - Instead of saving every crop to disk as a new image file,
      we store where each crop is and load it on the fly.
    - This saves time and disk space.
    """

    crop_list = []

    # Loop through all train images
    for img_path in sorted(TRAIN_IMAGES.glob("*.jpg")) + sorted(TRAIN_IMAGES.glob("*.png")):
        stem = img_path.stem

        # Matching label file for this image
        label_path = TRAIN_LABELS / (stem + ".txt")

        # Skip images with no label file
        if not label_path.exists():
            continue

        # Read image size
        try:
            w, h = Image.open(img_path).size
        except Exception:
            continue

        # Load GT boxes from label file
        gt_boxes = load_gt_boxes(label_path, w, h)

        # Build one crop entry per GT box
        for cls_id, xyxy in gt_boxes:
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

            # Box width/height
            bw, bh = x2 - x1, y2 - y1

            # Add padding so fine-tune crops look more like Step 3 crops
            # Source: expand box by fraction of its size (x1 -= bw*pad, x2 += bw*pad); same as step3_two_stage_inference.crop_box.
            x1 = x1 - bw * CROP_PAD
            y1 = y1 - bh * CROP_PAD
            x2 = x2 + bw * CROP_PAD
            y2 = y2 + bh * CROP_PAD

            # Clip again after padding
            # Source: same clipping as load_gt_boxes above; keeps crop inside image.
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Save crop metadata (not the crop image itself)
            crop_list.append((str(img_path), int(x1), int(y1), int(x2), int(y2), cls_id))

    return crop_list


class ThinSmearCropDataset(Dataset):
    """
    Dataset that loads thin-smear GT crops on the fly.

    Each item returned is:
      (crop_tensor, label)

    Why:
    - We don't want to save 24k crop files to disk
    - We can cut the crop directly from the original image when needed
    """

    def __init__(self, crop_list, transform):
        self.crop_list = crop_list
        self.transform = transform

    def __len__(self):
        return len(self.crop_list)

    def __getitem__(self, i):
        # Read crop information
        img_path, x1, y1, x2, y2, label = self.crop_list[i]

        # Load original image
        img = np.array(Image.open(img_path).convert("RGB"))

        # Cut crop from image
        crop = img[y1:y2, x1:x2]

        # Fallback in case crop is empty (very rare, but keeps dataset safe)
        if crop.size == 0:
            crop = np.zeros((1, 1, 3), dtype=np.uint8)

        # Convert crop to PIL image so torchvision transforms work
        pil = Image.fromarray(crop)

        # Return transformed crop + class label
        return self.transform(pil), label


def main():
    # ----------------------------
    # Safety checks
    # ----------------------------

    # Need the original 27k classifier first
    if not CLASSIFIER_27K.exists():
        print(f"ERROR: {CLASSIFIER_27K} not found. Run step2_train_classifier_27k.py first.")
        sys.exit(1)

    # Need processed train images and labels
    if not TRAIN_IMAGES.exists() or not TRAIN_LABELS.exists():
        print(f"ERROR: Train set not found: {TRAIN_IMAGES} or {TRAIN_LABELS}")
        sys.exit(1)

    # ----------------------------
    # Reproducibility
    # ----------------------------

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # ----------------------------
    # Build GT crop list
    # ----------------------------

    crop_list = build_crop_list()

    # If no crops were found, stop
    if len(crop_list) == 0:
        print("ERROR: No valid GT crops found in train set.")
        sys.exit(1)

    # ----------------------------
    # Split thin-smear crops into train/val
    # ----------------------------

    n_val = max(1, int(len(crop_list) * VAL_RATIO))
    n_train = len(crop_list) - n_val

    # random_split works on Dataset objects, so wrap crop_list in a simple dataset
    class ListWrapper(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    full_ds = ListWrapper(crop_list)

    # Reproducible train/val split
    train_subset, val_subset = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Convert split indices back into plain lists
    train_list = [crop_list[i] for i in train_subset.indices]
    val_list = [crop_list[i] for i in val_subset.indices]

    # ----------------------------
    # Image transforms
    # ----------------------------

    # Same normalization as Step 2
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transform = same idea as Step 2, with mild augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transform = no randomness
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # Build datasets
    train_dataset = ThinSmearCropDataset(train_list, train_transform)
    val_dataset = ThinSmearCropDataset(val_list, val_transform)

    # ----------------------------
    # DataLoaders
    # ----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ----------------------------
    # Load classifier from Step 2
    # ----------------------------

    # Load checkpoint from the 27k-only classifier
    ckpt = torch.load(CLASSIFIER_27K, map_location=device, weights_only=False)

    # Rebuild same architecture
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # Load trained weights
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Move to device
    model = model.to(device)

    # Training mode
    model.train()

    # Standard loss + optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Explicit class mapping
    class_to_idx = {"Parasitized": 0, "Uninfected": 1}

    # ----------------------------
    # Train / fine-tune loop
    # ----------------------------

    print("Step 2b: Fine-tune classifier on thin-smear GT crops")
    print("=" * 50)
    print(f"GT crops: {len(crop_list)} total, train: {n_train}, val: {n_val}")
    print(f"Device: {device}, Epochs: {EPOCHS}, LR: {LR}")
    print()

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ---- TRAIN PHASE ----
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()   # clear old gradients
            out = model(images)     # forward pass
            loss = criterion(out, labels)
            loss.backward()         # compute gradients
            optimizer.step()        # update weights

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION PHASE ----
        model.eval()
        correct = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                out = model(images)
                val_loss += criterion(out, labels).item()

                # Count correct predictions
                correct += (out.argmax(dim=1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_list)

        # ----------------------------
        # Save best fine-tuned model
        # ----------------------------

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": class_to_idx,
                "img_size": IMG_SIZE,
                "arch": "resnet18",
                "weights": "None",
                "finetuned_from": str(CLASSIFIER_27K),  # record original source model
            }, OUTPUT_DIR / "best.pt")

        # Print progress every 2 epochs and at epoch 1
        if epoch % 2 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"best={best_acc:.4f}"
            )

    # ----------------------------
    # Done
    # ----------------------------

    print()
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Saved: {OUTPUT_DIR / 'best.pt'}")
    print("Next: run Step 3 with --classifier_weights runs/classifier_27k_finetuned/best.pt")
    print("      then Step 4 to compare two-stage (27k) vs two-stage (27k+finetuned).")


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()