"""
Verify annotation conversion: draw bounding boxes on a sample image.
Run after convert_to_yolo.py to spot-check that boxes match cells.
Requires: pip install Pillow
"""

from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Install Pillow: pip install Pillow")
    raise

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def draw_boxes(img_path, label_path, output_path):
    """Draw YOLO boxes on image and save."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    colors = [(255, 0, 0), (0, 255, 0)]  # parasitized=red, uninfected=green

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            color = colors[int(cid)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    img.save(output_path)
    print(f"Saved: {output_path}")


def main():
    import sys

    train_imgs = DATA_DIR / "images" / "train"
    train_labels = DATA_DIR / "labels" / "train"

    if not train_imgs.exists():
        print("Run create_splits.py and convert_to_yolo.py first.")
        return

    output_dir = PROJECT_ROOT / "data" / "verify_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # If patient ID given (e.g. 247C99P60ThinF), verify that patient's images
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
        img_files = list(train_imgs.glob(f"{prefix}*.jpg"))
        if not img_files:
            print(f"No images found for {prefix}")
            return
    else:
        # Default: first 3 images + one with parasitized cells
        img_files = list(train_imgs.glob("*.jpg"))[:3]
        # Add 247C99P60ThinF (has parasitized cells) so you can see red boxes
        extra = list(train_imgs.glob("247C99P60ThinF*.jpg"))
        img_files = list(dict.fromkeys(img_files + extra[:1]))  # avoid duplicates

    for img_path in img_files:
        label_path = train_labels / (img_path.stem + ".txt")
        if label_path.exists():
            out_path = output_dir / f"verify_{img_path.name}"
            draw_boxes(img_path, label_path, out_path)

    print(f"\nCheck {output_dir} - red=parasitized, green=uninfected")


if __name__ == "__main__":
    main()
