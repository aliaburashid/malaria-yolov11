"""
Step 2: Train the Stage-2 classifier (CNN) on the 27k cell images.

Goal:
- Learn to classify a SINGLE cropped cell image as:
  0 = Parasitized
  1 = Uninfected

Why this exists:
- In the two-stage pipeline:
  Stage 1 (YOLO) finds cell boxes in a full smear image
  Stage 2 (this CNN) classifies each cropped cell box

What it does:
- Loads NIH cell images from:
    data/cell_images/Parasitized/
    data/cell_images/Uninfected/
- Splits into Train/Val (80/20) with fixed seed for reproducibility
- Trains ResNet18 (pretrained) for 2-class classification
- Saves the best model (highest val accuracy) to:
    runs/classifier_27k/best.pt

Run:
  python3 scripts/two_stage_baseline/step2_train_classifier_27k.py
"""

import sys
from pathlib import Path

# --- Paths: where the 27k cell images live and where we save the trained model ---
# PROJECT_ROOT: go up from this file (scripts/two_stage_baseline/) to repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Where the 27k NIH cell dataset lives (must contain Parasitized/ and Uninfected/
CELL_IMAGES_DIR = PROJECT_ROOT / "data" / "cell_images"   
# Where we will save the best trained classifier checkpoint
OUTPUT_DIR = PROJECT_ROOT / "runs" / "classifier_27k" 

# --- Training Constants: These define how the neural network will train.---
SEED = 42             # Fixed seed so If you run training again -> same split and same results. Without it results would change slightly each run.
VAL_RATIO = 0.2       # 20% of the dataset becomes validation; 80% becomes training
IMG_SIZE = 224        # ResNet expects 224x224; Step 3 will resize crops to this
BATCH_SIZE = 32       # How many images to process at once in training/validation   
EPOCHS = 20           # How many times the model sees the entire dataset. "Epoch 1 -> see all images once. Epoch 20 -> see all images twenty times."
LR = 1e-3             # Learning rate (Controls how fast the model updates its weights)


def main():
    # Import necessary libraries for training
    import torch #core neural network operations
    from torch.utils.data import DataLoader, Dataset, random_split # loads data in batches, load image dataset, split into train and val
    from torchvision import datasets, models, transforms # load ResNet, image preprocessing

    # Set random seeds so runs are reproducible (PyTorch: https://pytorch.org/docs/stable/notes/randomness.html)
    torch.manual_seed(SEED)
    # If using GPU, also seed GPU randomness
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Safety checks: dataset exists
    # If the dataset folder is missing, training cannot run, so stop early with a clear message
    if not CELL_IMAGES_DIR.exists():
        print(f"ERROR: {CELL_IMAGES_DIR} not found. Run step1_check_cell_images.py first.")
        sys.exit(1)

    # Create output folder so we can save best.pt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Image preprocessing
    # ----------------------------

    # Normalization values used for ImageNet-trained models (ResNet was trained with these)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Validation: only resize, convert to tensor, normalize with ImageNet stats (no randomness). 
    # Source: torchvision transforms.
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # Training: same as val plus mild augmentation (cells may appear rotated, flipped, or with different lighting)
    # so model generalizes to YOLO crops. 
    # Source: torchvision transforms.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # keep input size consistent
        transforms.RandomHorizontalFlip(p=0.5),   # prevent overfitting to orientation
        transforms.RandomRotation(15),   # mimic microscope rotation variation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # lighting variance
        transforms.ToTensor(),
        normalize,
    ])

    # ----------------------------
    # Load dataset (ImageFolder)
    # ----------------------------

    # ImageFolder reads:
    #   data/cell_images/Parasitized/*.png -> label 0 (alphabetical)
    #   data/cell_images/Uninfected/*.png  -> label 1 (alphabetical)
    #
    # We set transform=None here because we want:
    # train subset -> train_transform
    # val subset   -> val_transform
    # Source: torchvision ImageFolder.
    full_dataset = datasets.ImageFolder(root=str(CELL_IMAGES_DIR), transform=None)

    # Ensure folder-to-label mapping matches our expected convention:
    # 0 = Parasitized, 1 = Uninfected
    expected = {"Parasitized": 0, "Uninfected": 1}
    if full_dataset.class_to_idx != expected:
        raise ValueError(f"class_to_idx mismatch: {full_dataset.class_to_idx} (expected {expected})")
    
    # Dataset size checks
    n_total = len(full_dataset)
    if n_total == 0:
        raise ValueError("No images found under data/cell_images/.")
    
    # Compute split sizes
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    # If split becomes 0 by mistake, stop
    if n_val == 0 or n_train == 0:
        raise ValueError(f"Bad split sizes: n_train={n_train}, n_val={n_val}")
    
    # Split dataset into train/val with fixed seed so it is repeatable. Source: torch.utils.data.random_split.
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    # ----------------------------
    # Apply different transforms to train vs val
    # ----------------------------

    # random_split returns "Subset" objects pointing to the original dataset.
    # We wrap them so that each subset uses the correct transform.
    class SubsetWithTransform(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, i):
            # full_dataset returns (PIL_image, label)
            img, label = self.subset[i]
            # apply correct transform (train augmentation or val clean transform)
            return self.transform(img), label

    train_dataset = SubsetWithTransform(train_subset, train_transform)
    val_dataset = SubsetWithTransform(val_subset, val_transform)

    # ----------------------------
    # DataLoaders (batching)
    # ----------------------------

    # Choose CPU or GPU automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pin_memory helps speed when using GPU; harmless on CPU
    pin = device.type == "cuda"

    # Train loader:
    # shuffle=True so batches are in random order each epoch (better learning)
    # torch.utils.data.DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin
    )
    # Val loader:
    # shuffle=False because order doesn’t matter and we want stable evaluation
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ----------------------------
    # Build model (ResNet18)
    # ----------------------------

    # Load ResNet18 pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the final fully-connected layer:
    # Original: outputs 1000 classes
    # New: outputs 2 classes (Parasitized vs Uninfected)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    # Move model to device (CPU or GPU)
    model = model.to(device)
    # Loss function for multi-class classification (2 classes here)
    criterion = torch.nn.CrossEntropyLoss()
    # Optimizer updates the model weights based on gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----------------------------
    # Training loop
    # ----------------------------

    print("Step 2: Train Stage-2 classifier (27k cells)")
    print("=" * 50)
    print(f"Train: {n_train}, Val: {n_val}, Device: {device}")
    print()

    best_acc = 0.0

    # Run multiple epochs (each epoch = model sees whole training set once)
    for epoch in range(1, EPOCHS + 1):
        # Train mode: model updates weights
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            # Move batch to device
            images, labels = images.to(device), labels.to(device)
            # Clear old gradients from previous batch
            optimizer.zero_grad() 
            # Forward pass: model predicts logits
            out = model(images)  
            # Compute loss compared to true labels
            loss = criterion(out, labels)
            # Backward pass: compute gradients
            loss.backward()   
            # Update weights using Adam optimizer       
            optimizer.step()          
            train_loss += loss.item()
        
        # Average loss across batches
        train_loss /= len(train_loader)

        # ---- VALIDATION PHASE ----
        model.eval()                  # evaluation mode (no training updates)
        correct = 0
        val_loss = 0.0

        # No gradients needed in validation (faster, less memory)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                out = model(images)   # forward pass
                val_loss += criterion(out, labels).item()

                # Predicted class is the argmax of logits
                pred = out.argmax(dim=1)

                # Count correct predictions
                correct += (pred == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / n_val

        # ----------------------------
        # Save best model checkpoint
        # Source: torch.save.
        # ----------------------------

        # If this is the best validation accuracy so far, save weights
        if val_acc > best_acc:
            best_acc = val_acc
            out_path = OUTPUT_DIR / "best.pt"

            # Save everything Step 3 needs to rebuild and use the classifier
            torch.save({
                "model_state_dict": model.state_dict(),  # main weights
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": full_dataset.class_to_idx,
                "img_size": IMG_SIZE,
                "arch": "resnet18",
                "weights": "IMAGENET1K_V1",
            }, out_path)

            # Print progress occasionally (not every epoch to keep logs clean)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"best={best_acc:.4f}"
            )

    print()
    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Saved: {OUTPUT_DIR / 'best.pt'}")
    print("Next: python3 scripts/two_stage_baseline/step3_two_stage_inference.py")


if __name__ == "__main__":
    main()
