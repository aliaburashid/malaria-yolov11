"""
Verify annotation conversion: load a few images + their YOLO labels,
draw bounding boxes on a sample image with class labels (parasitized/uninfected).
Run after convert_to_yolo.py to spot-check that boxes match cells.
Requires: pip install Pillow

- PIL ImageDraw rectangle from corners:
  https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
"""

# build file paths safely
from pathlib import Path

# Try to import Pillow library (used for opening images and drawing on them)
try:
    from PIL import Image, ImageDraw
except ImportError:
    # If Pillow is not installed, tell the user how to install it
    print("Install Pillow: pip install Pillow")
    raise

# Get project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Get processed data folder
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def draw_boxes(img_path, label_path, output_path):
    """
    Draw YOLO bounding boxes on image and save it.

    Parameters:
    - img_path: path to original image
    - label_path: path to YOLO .txt label file
    - output_path: where to save the new image with boxes
    """
    # Open image and force RGB format (important for drawing colors)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size # Get image width and height
    draw = ImageDraw.Draw(img) # Create drawing object to draw rectangles on image

    # Define colors for classes:
    colors = [(255, 0, 0), (0, 255, 0)]  # parasitized=red, uninfected=green

    # Open the YOLO label file
    # Each line looks like:
    # class_id  x_center  y_center  width  height
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            # If something is wrong (not exactly 5 numbers), skip it
            if len(parts) != 5:
                continue

            # Convert values from string → float
            # cid = class id (0 or 1)
            # xc = normalized x center
            # yc = normalized y center
            # bw = normalized box width
            # bh = normalized box height
            cid, xc, yc, bw, bh = map(float, parts)
           
            # YOLO format is normalized 0–1; multiply by w/h for pixels.
            # Convert center-based (xc,yc,bw,bh) to corner-based (x1,y1,x2,y2).
            # Refs: https://docs.ultralytics.com/datasets/detect/ ;
            #       https://stackoverflow.com/questions/56115874/ (same formulas)
            x1 = (xc - bw / 2) * w   # top-left x
            y1 = (yc - bh / 2) * h   # top-left y
            x2 = (xc + bw / 2) * w   # bottom-right x
            y2 = (yc + bh / 2) * h   # bottom-right y
            # Choose color based on class ID
            color = colors[int(cid)]
            # Draw rectangle on image
            # width=3 means border thickness
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Save new image with bounding boxes drawn
    img.save(output_path)
    print(f"Saved: {output_path}")


def main():
    import sys

    # Define where training images are stored
    train_imgs = DATA_DIR / "images" / "train"
    # Define where training label files are stored
    train_labels = DATA_DIR / "labels" / "train"

    # Check if training images folder exists
    # If not, dataset hasn’t been prepared yet
    if not train_imgs.exists():
        print("Run create_splits.py and convert_to_yolo.py first.")
        return
    
    # Create folder to store verification results
    # project/data/verify_samples
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
        # Take first 3 images in training set
        img_files = list(train_imgs.glob("*.jpg"))[:3]
        # Add 247C99P60ThinF (has parasitized cells) so you can see red boxes
        extra = list(train_imgs.glob("247C99P60ThinF*.jpg"))
        img_files = list(dict.fromkeys(img_files + extra[:1]))  # avoid duplicates
    
    # process each image
    for img_path in img_files:
        # Find corresponding label file
        label_path = train_labels / (img_path.stem + ".txt")
        # If label file exists, draw boxes and save to verify_samples folder
        if label_path.exists():
            # Create output path for verification image
            out_path = output_dir / f"verify_{img_path.name}"
            # Draw boxes on image and save to output path
            draw_boxes(img_path, label_path, out_path)

    print(f"\nCheck {output_dir} - red=parasitized, green=uninfected")


if __name__ == "__main__":
    main()
