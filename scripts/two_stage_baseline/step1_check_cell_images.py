"""
Step 1: Check that the 27k cell dataset is present and correctly laid out.

Counts images in data/cell_images/Parasitized/ and data/cell_images/Uninfected/,
prints a short summary, and exits with "OK" when both folders exist and contain images.

Run from project root: python3 scripts/two_stage_baseline/step1_check_cell_images.py

References:
- NIH malaria cell imagery (27,558 cells): https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html
- Option A two-stage pipeline: docs/OPTION_A_TWO_STAGE.md
"""

import sys
from pathlib import Path

# Resolve project root (this file lives in scripts/two_stage_baseline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Expected location of the 27k cell dataset (Parasitized + Uninfected subfolders)
CELL_IMAGES_DIR = PROJECT_ROOT / "data" / "cell_images"
PARASITIZED_DIR = CELL_IMAGES_DIR / "Parasitized"
UNINFECTED_DIR = CELL_IMAGES_DIR / "Uninfected"

# Extensions accepted as cell images (NIH dataset is typically PNG)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def count_images(folder: Path) -> int:
    """Count image files in a folder (only files with IMAGE_EXTENSIONS)."""
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    # Print header and path we are checking
    print("Step 1: Check 27k cell dataset")
    print("=" * 50)
    print(f"Looking for: {CELL_IMAGES_DIR}")
    print()

    # Fail if the dataset directory is missing (user must download and unzip)
    if not CELL_IMAGES_DIR.exists():
        print("NOT FOUND: data/cell_images/ does not exist.")
        print("Download cell_images.zip from NIH malaria datasheet and unzip into data/cell_images/")
        print("Expected: data/cell_images/Parasitized/ and data/cell_images/Uninfected/")
        sys.exit(1)

    # Count images in each class folder (reference: ~13,779 per class, ~27,558 total)
    n_parasitized = count_images(PARASITIZED_DIR)
    n_uninfected = count_images(UNINFECTED_DIR)
    total = n_parasitized + n_uninfected

    # Report counts to the user
    print(f"  Parasitized: {n_parasitized} images")
    print(f"  Uninfected:  {n_uninfected} images")
    print(f"  Total:      {total} images")
    print()

    # Require at least some images in both classes
    if total == 0:
        print("No images found. Check that Parasitized/ and Uninfected/ contain .png or .jpg files.")
        sys.exit(1)

    if n_parasitized == 0 or n_uninfected == 0:
        print("WARNING: One class has no images. Both Parasitized and Uninfected should have images.")
        sys.exit(1)

    # Soft check: NIH 27k set is ~27,558 images; warn if much lower (incomplete unzip?)
    if total < 20_000:
        print("WARNING: Total image count seems lower than expected (~27k). Check unzip.")
        print()

    # Success: dataset ready for Step 2 (train classifier)
    print("OK — Dataset ready for Step 2. Run: python3 scripts/two_stage_baseline/step2_train_classifier_27k.py")
    sys.exit(0)


if __name__ == "__main__":
    main()
