"""
Step 1: Create corrupted copies of the test set for robustness evaluation.

Corruptions (each with mild / medium / strong):
- Blur: out-of-focus (Gaussian blur).
- Brightness: darker images.
- Contrast: weak staining / faded (lower contrast).
- Noise: sensor / grainy (additive Gaussian noise).
- JPEG: lower quality (JPEG compression).

Labels are copied unchanged from data/processed/labels/test.
Same image names so evaluation scripts can match predictions to GT.

Run from project root:
  python3 scripts/robustness/step1_create_corrupted_test_sets.py

Sources / notes:
- Corruption families are aligned with common robustness benchmarks
  (e.g., blur, brightness/contrast shift, Gaussian noise, JPEG artifacts).
- Parameters are centralized in corruption_definitions.py so Figure 4 and
  robustness runs use the same settings.
"""

# Used to copy label files while preserving metadata.
import shutil
from pathlib import Path
from typing import Optional

# Reference (reproducibility practice): fixed random seed for deterministic outputs.
# See: NumPy random generation docs (https://numpy.org/doc/stable/reference/random/)
# Used for deterministic behavior where random noise is involved.
import numpy as np
# PIL handles image loading/saving and basic image operations.
from PIL import Image

# Reference (corruption families): common corruption robustness benchmarks.
# See: Hendrycks and Dietterich (2019), "Benchmarking Neural Network Robustness..."
# Reference (medical adaptation context): Di Salvo et al. (2024), MedMNIST-C.
# Import shared corruption presets and helper functions.
from corruption_definitions import (
    CORRUPTIONS,
    corrupt_blur,
    corrupt_brightness,
    corrupt_contrast,
    corrupt_jpeg,
    corrupt_noise,
)

# Project root (this file is in scripts/robustness/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Source test images created by preprocessing pipeline.
TEST_IMAGES = PROJECT_ROOT / "data" / "processed" / "images" / "test"
# Source YOLO labels for the test split.
TEST_LABELS = PROJECT_ROOT / "data" / "processed" / "labels" / "test"
# Output root where clean + corrupted copies are written.
OUT_ROOT = PROJECT_ROOT / "data" / "processed_corrupted"


def build_corrupted_set(name: str, apply_corrupt, save_as_jpeg_quality: Optional[int] = None):
    """Build one corrupted test set: images in OUT_ROOT/name/images/test, labels in OUT_ROOT/name/labels/test.

    If save_as_jpeg_quality is set (e.g. for jpeg_*), save all images as .jpg with that quality
    so JPEG artifact effect is visible even when originals are PNG.
    """
    # Output folder for images under this condition.
    img_out = OUT_ROOT / name / "images" / "test"
    # Output folder for labels under this condition.
    label_out = OUT_ROOT / name / "labels" / "test"
    # Create image output directory (and parents) if missing.
    img_out.mkdir(parents=True, exist_ok=True)
    # Create label output directory (and parents) if missing.
    label_out.mkdir(parents=True, exist_ok=True)

    # Process all test images (.jpg and .png) in a deterministic order.
    image_files = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
    # Iterate through each test image and build a corrupted counterpart.
    for path in image_files:
        # Open image file with PIL.
        with Image.open(path) as im:
            # Force RGB mode so all corruption functions get consistent input.
            img = im.convert("RGB")
        # Apply the selected corruption function for this condition.
        corrupted = apply_corrupt(img)
    # Reference (JPEG artifact simulation): image saved with controlled JPEG quality.
    # This is the standard way to introduce compression artifacts in robustness tests.
    # For jpeg_* conditions, force JPEG output at the selected quality.
        if save_as_jpeg_quality is not None:
            # Ensure extension is .jpg when testing JPEG artifacts.
            out_path = img_out / (path.stem + ".jpg")
            # Save with explicit JPEG quality to introduce compression artifacts.
            corrupted.save(out_path, format="JPEG", quality=save_as_jpeg_quality)
        else:
            # Keep original filename/extension for non-JPEG conditions.
            out_path = img_out / path.name
            # Save PNGs losslessly.
            if out_path.suffix.lower() == ".png":
                corrupted.save(out_path, format="PNG")
            else:
                # Save JPEGs with high quality to avoid adding extra artifacts.
                corrupted.save(out_path, format="JPEG", quality=95)

        # Copy corresponding YOLO label file (same stem, .txt extension).
        label_src = TEST_LABELS / (path.stem + ".txt")
        # Only copy if label exists (defensive check).
        if label_src.exists():
            shutil.copy2(label_src, label_out / (path.stem + ".txt"))

    # Reference (Ultralytics dataset format): dataset.yaml requires path/train/val/test/names/nc.
    # See: Ultralytics docs - dataset configuration for detection tasks.
    # Write minimal dataset.yaml so YOLO can validate directly on this folder.
    yaml_path = OUT_ROOT / name / "dataset.yaml"
    # Resolve absolute path to avoid ambiguity when script is run from anywhere.
    abs_path = yaml_path.parent.resolve()
    # Point val/test at images/test; train is created empty below.
    yaml_path.write_text(
        f"path: {abs_path}\n"
        "train: images/train\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n  0: parasitized\n  1: uninfected\nnc: 2\n"
    )
    # Create empty train dirs so YOLO config remains structurally valid.
    (OUT_ROOT / name / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / name / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Return number of processed images for summary printing.
    return len(image_files)


def main():
    # Reference (experimental reproducibility): fixed seed ensures same noise draws.
    # Seed NumPy RNG so any random noise operation is reproducible.
    np.random.seed(42)
    # Stop early if expected processed test data is missing.
    if not TEST_IMAGES.exists() or not TEST_LABELS.exists():
        print(f"Test set not found: {TEST_IMAGES} / {TEST_LABELS}")
        print("Run scripts/class_imbalance/create_splits.py and convert_to_yolo.py first.")
        return

    # Ensure top-level output directory exists.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Running total of images produced across all conditions.
    total = 0
    # Build clean copy first; later steps expect this condition to exist too.
    print("  clean: ", end="")
    # Identity corruption: write unchanged images + copied labels.
    n = build_corrupted_set("clean", lambda img: img)
    # Update total counter.
    total += n
    # Print per-condition image count.
    print(f"{n} images")

    # Loop through each corruption family and each intensity level.
    for corruption, levels in CORRUPTIONS.items():
        for level, params in levels.items():
            # Condition folder name (e.g., blur_mild, noise_strong).
            name = f"{corruption}_{level}"
            # Reference (single-source parameter control): corruption intensities are read
            # from CORRUPTIONS in corruption_definitions.py to keep Figure 4 and robustness
            # experiment settings identical.
            # Select corruption function and optional JPEG quality behavior.
            if corruption == "blur":
                apply = lambda img, p=params: corrupt_blur(img, p["radius"])
                jpeg_q = None
            elif corruption == "brightness":
                apply = lambda img, p=params: corrupt_brightness(img, p["factor"])
                jpeg_q = None
            elif corruption == "contrast":
                apply = lambda img, p=params: corrupt_contrast(img, p["factor"])
                jpeg_q = None
            elif corruption == "noise":
                apply = lambda img, p=params: corrupt_noise(img, p["std"])
                jpeg_q = None
            elif corruption == "jpeg":
                apply = lambda img, p=params: corrupt_jpeg(img, p["quality"])
                jpeg_q = params["quality"]
            else:
                # Ignore unknown corruption keys safely.
                continue
            # Build this corrupted condition on disk.
            n = build_corrupted_set(name, apply, save_as_jpeg_quality=jpeg_q)
            # Update global total.
            total += n
            # Print condition-level summary for progress visibility.
            print(f"  {name}: {n} images")

    # Final status summary and next step pointer.
    print(f"Done. Clean + corrupted sets in {OUT_ROOT}")
    print("Next: python3 scripts/robustness/step2_run_yolo_robustness.py")


# Standard Python entry point guard.
if __name__ == "__main__":
    main()
