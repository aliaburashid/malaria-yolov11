"""
Step 2: Train the Stage-2 classifier (CNN) on the 27k cell images.

Loads data/cell_images/Parasitized and Uninfected, splits train/val (80/20),
trains a CNN (parasitized vs uninfected), saves the best model to runs/classifier_27k/.

Run from project root: python3 scripts/two_stage_baseline/step2_train_classifier_27k.py
Sources are cited in comments directly above each relevant code block.
"""

import sys
from pathlib import Path

# --- Paths: where the 27k cell images live and where we save the trained model ---
# PROJECT_ROOT: go up from this file (scripts/two_stage_baseline/) to repo root. Data path: NIH 27k cells (Option A).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CELL_IMAGES_DIR = PROJECT_ROOT / "data" / "cell_images"   # Parasitized/ and Uninfected/ subfolders
OUTPUT_DIR = PROJECT_ROOT / "runs" / "classifier_27k"     # best.pt will be saved here

# --- Constants: same seed as rest of project; 80% train / 20% val; image size for ResNet; training hyperparams ---
SEED = 42
VAL_RATIO = 0.2       # 20% of data for validation
IMG_SIZE = 224        # ResNet expects 224x224; Step 3 will resize crops to this
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3


def main():
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import datasets, models, transforms

    # Set random seeds so runs are reproducible (PyTorch: https://pytorch.org/docs/stable/notes/randomness.html)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Exit if 27k dataset is missing (user must run step1 and put cell_images in data/)
    if not CELL_IMAGES_DIR.exists():
        print(f"ERROR: {CELL_IMAGES_DIR} not found. Run step1_check_cell_images.py first.")
        sys.exit(1)

    # Create output directory for best.pt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validation: only resize, convert to tensor, normalize with ImageNet stats (no randomness). Source: torchvision transforms.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    # Training: same as val plus mild augmentation (flip, rotation, color jitter) so model generalizes to YOLO crops. Source: torchvision transforms.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    # Load all images: ImageFolder expects cell_images/Parasitized/*.png and Uninfected/*.png. transform=None so we apply train/val transform per split. Source: torchvision ImageFolder.
    full_dataset = datasets.ImageFolder(root=str(CELL_IMAGES_DIR), transform=None)
    # Ensure class IDs match YOLO (0=parasitized, 1=uninfected); ImageFolder assigns by alphabetical folder name.
    expected = {"Parasitized": 0, "Uninfected": 1}
    if full_dataset.class_to_idx != expected:
        raise ValueError(f"class_to_idx mismatch: {full_dataset.class_to_idx} (expected {expected})")
    n_total = len(full_dataset)
    if n_total == 0:
        raise ValueError("No images found under data/cell_images/.")
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val
    if n_val == 0 or n_train == 0:
        raise ValueError(f"Bad split sizes: n_train={n_train}, n_val={n_val}")
    # Split into train and val indices with fixed seed so the split is reproducible. Source: torch.utils.data.random_split.
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    # Wrapper so train gets augmentation, val gets resize+normalize only (custom; Dataset API from PyTorch data)
    class SubsetWithTransform(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, i):
            img, label = self.subset[i]
            return self.transform(img), label

    train_dataset = SubsetWithTransform(train_subset, train_transform)
    val_dataset = SubsetWithTransform(val_subset, val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    # torch.utils.data.DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Build model: ResNet18 pretrained on ImageNet, replace final layer with 2 outputs (parasitized, uninfected). Move to device. Loss and optimizer for training. Source: torchvision models, PyTorch nn/optim.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Step 2: Train Stage-2 classifier (27k cells)")
    print("=" * 50)
    print(f"Train: {n_train}, Val: {n_val}, Device: {device}")
    print()

    best_acc = 0.0
    # Epoch loop: train mode, forward pass, loss backward, optimizer step; then eval mode, no grad, compute val loss and accuracy.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()      # clear gradients from previous batch
            out = model(images)        # forward pass
            loss = criterion(out, labels)
            loss.backward()            # compute gradients
            optimizer.step()           # update weights
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct = 0
        val_loss = 0.0
        with torch.no_grad():          # no gradients needed for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_loss += criterion(out, labels).item()
                pred = out.argmax(dim=1)   # predicted class (0 or 1)
                correct += (pred == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = correct / n_val

        # If this epoch has the best val accuracy so far, save checkpoint (state_dict + metadata for Step 3). Source: torch.save.
        if val_acc > best_acc:
            best_acc = val_acc
            out_path = OUTPUT_DIR / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),   # weights to load in Step 3
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": full_dataset.class_to_idx,
                "img_size": IMG_SIZE,
                "arch": "resnet18",
                "weights": "IMAGENET1K_V1",
            }, out_path)

        # Print progress every 5 epochs and at epoch 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  best={best_acc:.4f}")

    print()
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Saved: {OUTPUT_DIR / 'best.pt'}")
    print("Next: python3 scripts/two_stage_baseline/step3_two_stage_inference.py")


if __name__ == "__main__":
    main()
