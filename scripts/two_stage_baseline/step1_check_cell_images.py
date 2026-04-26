"""
Step 1: Check that the 27k cell dataset is present and correctly laid out.

Counts images in:
- data/cell_images/Parasitized/
- data/cell_images/Uninfected/

Prints a summary, and exits with "OK" when both folders exist and contain images.

Run from project root: python3 scripts/two_stage_baseline/step1_check_cell_images.py

"""
# sys is used so we can exit the script with success/failure codes (0 = OK, 1 = error)
import sys
from pathlib import Path 

# PROJECT_ROOT = the top of your repo.
# __file__ = this script file path.
# .resolve() makes it an absolute path.
# parent.parent.parent goes up 3 folders because this file is in scripts/two_stage_baseline/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Expected location of the 27k cell dataset (Parasitized + Uninfected subfolders)
CELL_IMAGES_DIR = PROJECT_ROOT / "data" / "cell_images"

# These two folders must exist inside cell_images/
# because the dataset is organized by class.
PARASITIZED_DIR = CELL_IMAGES_DIR / "Parasitized"
UNINFECTED_DIR = CELL_IMAGES_DIR / "Uninfected"

# Only count real image files with these extensions.
# NIH is usually PNG, but we allow common formats too.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def count_images(folder: Path) -> int:
    """Count image files in a folder (only files with IMAGE_EXTENSIONS)."""

    # If the folder path doesn't exist or isn't a directory,
    # we return 0 so the caller can detect a missing folder
    if not folder.is_dir():
        return 0
    
    # folder.iterdir() lists items directly inside the folder.
    # We count only those whose file extension matches IMAGE_EXTENSIONS.
    # .suffix gives extension like ".png"
    # .lower() makes it case-insensitive (".PNG" -> ".png")
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    # Print header and path we are checking
    print("Step 1: Check 27k cell dataset")
    print("=" * 50)

    # Show exactly what path we are checking to avoid confusion
    print(f"Looking for: {CELL_IMAGES_DIR}")
    print()

    # If the user didn't download/unzip the dataset,
    # CELL_IMAGES_DIR won't exist and Step 2 would fail.
    if not CELL_IMAGES_DIR.exists():
        print("NOT FOUND: data/cell_images/ does not exist.")
        print("Download cell_images.zip from NIH malaria datasheet and unzip into data/cell_images/")
        print("Expected: data/cell_images/Parasitized/ and data/cell_images/Uninfected/")
        # Exit with failure code (1) so the user knows to fix it
        sys.exit(1)

    # Count how many images exist in each class folder.
    # This checks that the unzip worked and the folder structure is correct.
    n_parasitized = count_images(PARASITIZED_DIR)
    n_uninfected = count_images(UNINFECTED_DIR)

    # Total number of images across both classes
    total = n_parasitized + n_uninfected

    # Print the results so the user can quickly confirm everything looks right
    print(f"  Parasitized: {n_parasitized} images")
    print(f"  Uninfected:  {n_uninfected} images")
    print(f"  Total:      {total} images")
    print()

    # If total is 0, then either:
    # - folders are empty,
    # - wrong extensions,
    # - dataset not unzipped properly
    if total == 0:
        print("No images found. Check that Parasitized/ and Uninfected/ contain .png or .jpg files.")
        sys.exit(1)
    
    # If one class has 0 images, training becomes broken/unfair.
    # This usually means the unzip or folder names are wrong.
    if n_parasitized == 0 or n_uninfected == 0:
        print("WARNING: One class has no images. Both Parasitized and Uninfected should have images.")
        sys.exit(1)

    # The NIH dataset normally has ~27,558 images total.
    # If we see much less than ~27k, then probably downloaded the wrong thing
    # or the unzip didn’t complete fully.
    if total < 20_000:
        print("WARNING: Total image count seems lower than expected (~27k). Check unzip.")
        print()

    # If we got here, the dataset is present and usable.
    # Print the next command so the workflow is super clear.
    print("OK — Dataset ready for Step 2. Run: python3 scripts/two_stage_baseline/step2_train_classifier_27k.py")
    # Exit with 0 to signal success
    sys.exit(0)

# This makes sure main() runs only when you execute this file directly,
# not when it’s imported from another script.
if __name__ == "__main__":
    main()
