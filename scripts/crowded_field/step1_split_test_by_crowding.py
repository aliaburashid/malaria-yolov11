"""
Step 1 — split test images into crowded vs sparse subsets using GT cell counts.

Method:
- Count non-empty lines in each YOLO label file under data/processed/labels/test.
- Use the median count as the split threshold.
- Write absolute image-path lists to data/splits/test_crowded.txt and
  data/splits/test_sparse.txt.

Reference:
- scripts/crowded_field/README.md (subset definitions and run order)

Run from project root:
  python3 scripts/crowded_field/step1_split_test_by_crowding.py
"""

from __future__ import annotations

# Standard library median for split threshold.
import statistics
# Path handling independent of current working directory.
from pathlib import Path
# Optional return for image lookup helper.
from typing import Optional

# Repository root anchor: this file is scripts/crowded_field/, so parents[2] is project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Input YOLO labels and images from processed test split.
LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "test"
IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "test"
# Output split lists consumed by step2_yolo_val_subsets.py.
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_CROWDED = SPLITS_DIR / "test_crowded.txt"
OUT_SPARSE = SPLITS_DIR / "test_sparse.txt"


def count_gt_lines(label_path: Path) -> int:
    # Count non-empty YOLO label rows = number of annotated cells.
    n = 0
    with open(label_path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def find_image_stem(stem: str) -> Optional[str]:
    # Match common test image extensions for the same label stem.
    for ext in (".jpg", ".png", ".jpeg"):
        p = IMAGE_DIR / (stem + ext)
        if p.exists():
            return stem + ext
    return None


def main() -> None:
    # Fail fast if dataset layout is missing.
    if not LABEL_DIR.is_dir():
        raise SystemExit(f"Missing label dir: {LABEL_DIR}")

    # Collect (absolute_image_path, gt_cell_count) rows used for median split.
    rows: list[tuple[str, int]] = []
    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        stem = label_path.stem
        img_name = find_image_stem(stem)
        if img_name is None:
            print(f"Warning: no image for label stem {stem}, skipping")
            continue
        c = count_gt_lines(label_path)
        # Use absolute paths so Ultralytics test-list loading is unambiguous.
        # Reference: scripts/crowded_field/README.md (split file format).
        abs_img = (IMAGE_DIR / img_name).resolve()
        rows.append((str(abs_img), c))

    # Abort if no valid label-image pairs were found.
    if not rows:
        raise SystemExit("No labelled test images found.")

    # Median threshold matches crowded/sparse definition in README.
    counts = [c for _, c in rows]
    median_c = statistics.median(counts)

    # Crowded: count >= median; Sparse: count < median.
    crowded = [p for p, c in rows if c >= median_c]
    sparse = [p for p, c in rows if c < median_c]

    # Write deterministic sorted lists for reproducible subset evaluation.
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CROWDED.write_text("\n".join(sorted(crowded)) + "\n")
    OUT_SPARSE.write_text("\n".join(sorted(sparse)) + "\n")

    print("Crowded-field Step 1 — test split by GT cell count")
    print(f"  Label dir:     {LABEL_DIR}")
    print(f"  Images:        {len(rows)}")
    print(f"  Median count:  {median_c}")
    print(f"  Crowded (>=):  {len(crowded)}  -> {OUT_CROWDED.relative_to(PROJECT_ROOT)}")
    print(f"  Sparse (<):    {len(sparse)}  -> {OUT_SPARSE.relative_to(PROJECT_ROOT)}")
    print("  Next: python3 scripts/crowded_field/step2_yolo_val_subsets.py")


if __name__ == "__main__":
    main()
