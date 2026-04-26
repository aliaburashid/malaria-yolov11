"""
Verify annotation conversion: load a few images + their YOLO labels,
draw bounding boxes on a sample image with class labels (parasitized/uninfected).
Run after convert_to_yolo.py to spot-check that boxes match cells.
Requires: pip install Pillow

- PIL ImageDraw rectangle from corners:
  https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
"""

from __future__ import annotations

# build file paths safely
from pathlib import Path

# Try to import Pillow library (used for opening images and drawing on them)
try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError:
    # If Pillow is not installed, tell the user how to install it
    print("Install Pillow: pip install Pillow")
    raise

# Get project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Get processed data folder
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _default_yolo_weights() -> Path | None:
    """First best.pt under runs/detect/, if any."""
    root = PROJECT_ROOT / "runs" / "detect"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("**/weights/best.pt"))
    return candidates[0] if candidates else None


def draw_boxes(
    img_path: Path,
    label_path: Path,
    output_path: Path,
    *,
    show_labels: bool = True,
) -> None:
    """
    Draw YOLO bounding boxes on image and save it.

    Parameters:
    - img_path: path to original image
    - label_path: path to YOLO .txt label file
    - output_path: where to save the new image with boxes
    - show_labels: if True, draw "p"/"u" above each box; if False, boxes only (red/green)
    """
    # Open image and force RGB format (important for drawing colors)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size # Get image width and height
    draw = ImageDraw.Draw(img) # Create drawing object to draw rectangles on image

    # Define colors for classes:
    colors = [(255, 0, 0), (0, 255, 0)]  # parasitized=red, uninfected=green
    class_letters = ("p", "u")

    font = None
    font_size = 0
    if show_labels:
        font_size = max(14, int(min(w, h) * 0.018))
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", size=font_size
            )
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    size=font_size,
                )
            except OSError:
                font = ImageFont.load_default()

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
            ci = int(cid)
            color = colors[ci]
            letter = class_letters[ci]
            # Draw rectangle on image
            # width=3 means border thickness
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            if show_labels and font is not None:
                tx, ty = x1 + 2, max(0.0, y1 - font_size - 4)
                for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                    draw.text(
                        (tx + ox, ty + oy),
                        letter,
                        fill=(0, 0, 0),
                        font=font,
                    )
                draw.text((tx, ty), letter, fill=color, font=font)

    # Save new image with bounding boxes drawn
    img.save(output_path)
    print(f"Saved: {output_path}")


def _yolo_boxes_by_class(
    label_path: Path, img_w: int, img_h: int
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Return (parasitized boxes, uninfected boxes) in pixel coords (x1,y1,x2,y2)."""
    parasitized: list[tuple[float, float, float, float]] = []
    uninfected: list[tuple[float, float, float, float]] = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            if int(cid) == 0:
                parasitized.append((x1, y1, x2, y2))
            else:
                uninfected.append((x1, y1, x2, y2))
    return parasitized, uninfected


def _iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _yolo_match_confidences(
    img_path: Path,
    weights: Path | None,
    selections: list[tuple[int, tuple[float, float, float, float]]],
    iou_min: float = 0.25,
) -> list[float | None]:
    """
    Run detector on full image; for each GT box, report detector confidence.

    Prefer the detection with the same class as GT and highest IoU. If the model
    disagrees on class (common for hard infected cells), fall back to the
    highest-IoU detection of any class on that location so a numeric
    confidence still reflects "how sure the detector is about a box here".
    """
    if weights is None or not weights.is_file():
        return [None] * len(selections)
    try:
        from ultralytics import YOLO
    except ImportError:
        return [None] * len(selections)

    model = YOLO(str(weights))
    r = model.predict(str(img_path), conf=0.001, verbose=False, imgsz=1280)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [None] * len(selections)

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    out: list[float | None] = []
    for cid, gt in selections:
        best_same_iou = 0.0
        best_same_c: float | None = None
        best_any_iou = 0.0
        best_any_c: float | None = None
        for k in range(len(xyxy)):
            iou = _iou_xyxy(gt, tuple(xyxy[k].tolist()))
            c = float(confs[k])
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_c = c
            if int(clss[k]) != cid:
                continue
            if iou > best_same_iou:
                best_same_iou = iou
                best_same_c = c
        if best_same_iou >= iou_min and best_same_c is not None:
            out.append(best_same_c)
        elif best_any_iou >= iou_min and best_any_c is not None:
            out.append(best_any_c)
        else:
            out.append(None)
    return out


def _expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    """Pad box by pad_frac of its width/height; clip to image."""
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw2 = bw * (1.0 + 2.0 * pad_frac)
    bh2 = bh * (1.0 + 2.0 * pad_frac)
    nx1 = max(0.0, cx - bw2 / 2)
    ny1 = max(0.0, cy - bh2 / 2)
    nx2 = min(float(img_w), cx + bw2 / 2)
    ny2 = min(float(img_h), cy + bh2 / 2)
    return int(nx1), int(ny1), int(nx2), int(ny2)


def _four_cell_tiles_and_selections(
    img_path: Path,
    label_path: Path,
    *,
    pad_frac: float = 0.12,
    target_h: int = 200,
    border: int = 3,
) -> tuple[list[Image.Image], list[tuple[int, tuple[float, float, float, float]]]]:
    """
    Build four PIL tiles (2 infected, 2 uninfected) and return GT boxes for IoU matching.
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    parasitized, uninfected = _yolo_boxes_by_class(label_path, W, H)
    if len(parasitized) < 2 or len(uninfected) < 2:
        raise ValueError(
            f"Need ≥2 parasitized and ≥2 uninfected boxes; got p={len(parasitized)} u={len(uninfected)}"
        )
    selections: list[tuple[int, tuple[float, float, float, float]]] = [
        (0, parasitized[0]),
        (0, parasitized[1]),
        (1, uninfected[0]),
        (1, uninfected[1]),
    ]
    border_rgb = [(255, 0, 0), (0, 255, 0)]
    tiles: list[Image.Image] = []
    for cid, box in selections:
        x1, y1, x2, y2 = box
        nx1, ny1, nx2, ny2 = _expand_box(x1, y1, x2, y2, W, H, pad_frac)
        crop = img.crop((nx1, ny1, nx2, ny2))
        cw, ch = crop.size
        if ch < 1 or cw < 1:
            raise RuntimeError("Empty crop")
        new_w = max(1, int(cw * target_h / ch))
        crop = crop.resize((new_w, target_h), Image.LANCZOS)
        crop = ImageOps.expand(crop, border=border, fill=border_rgb[cid])
        tiles.append(crop)
    return tiles, selections


def save_four_crop_row(
    img_path: Path,
    label_path: Path,
    output_path: Path,
    *,
    pad_frac: float = 0.12,
    target_h: int = 200,
    gap: int = 12,
    border: int = 3,
) -> None:
    """
    One row: 2 parasitized crops + 2 uninfected crops (GT boxes), red/green borders.
    Uses first two boxes of each class in label file order.
    """
    tiles, _ = _four_cell_tiles_and_selections(
        img_path, label_path, pad_frac=pad_frac, target_h=target_h, border=border
    )
    total_w = sum(t.size[0] for t in tiles) + gap * (len(tiles) - 1)
    out = Image.new("RGB", (total_w, tiles[0].size[1]), (255, 255, 255))
    x_off = 0
    for t in tiles:
        out.paste(t, (x_off, 0))
        x_off += t.size[0] + gap
    out.save(output_path)
    print(f"Saved: {output_path}")


def save_four_crop_row_academic(
    img_path: Path,
    label_path: Path,
    output_path: Path,
    *,
    weights: Path | None = None,
    pad_frac: float = 0.12,
    target_h: int = 220,
    dpi: int = 300,
) -> None:
    """
    Four crops in one row with class text (Parasitized / Uninfected) and p = … under each
    (detector confidence; YOLO match by IoU, see _yolo_match_confidences).
    """
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError:
        raise ImportError("Install matplotlib for --academic: pip install matplotlib")

    tiles, selections = _four_cell_tiles_and_selections(
        img_path, label_path, pad_frac=pad_frac, target_h=target_h, border=3
    )
    confs = _yolo_match_confidences(img_path, weights, selections)

    class_title = ("Parasitized", "Uninfected")
    fig_w = 4 * 2.2
    fig_h = 3.1
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        4,
        height_ratios=[1.0, 0.22],
        hspace=0.35,
        wspace=0.18,
        left=0.05,
        right=0.95,
        top=0.94,
        bottom=0.08,
    )
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.asarray(tiles[i]))
        ax.set_axis_off()
        cid = selections[i][0]
        cstr = (
            f"{confs[i]:.3f}"
            if confs[i] is not None
            else "—"
        )
        ax_txt = fig.add_subplot(gs[1, i])
        ax_txt.set_axis_off()
        ax_txt.text(
            0.5,
            0.72,
            class_title[cid],
            transform=ax_txt.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1a1a1a",
        )
        ax_txt.text(
            0.5,
            0.28,
            f"p = {cstr}",
            transform=ax_txt.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#333333",
        )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw GT boxes on training images for conversion QA."
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        default=None,
        help="Only images whose filenames start with this (e.g. 247C99P60ThinF). "
        "If omitted, uses first 3 train images plus one 247C99P60ThinF sample.",
    )
    parser.add_argument(
        "--boxes-only",
        action="store_true",
        help="Red/green boxes only (no p/u text). Saves as verify_boxes_<name>.jpg",
    )
    parser.add_argument(
        "--crop-row",
        metavar="STEM",
        default=None,
        help="Save one row of 4 crops (2 parasitized, 2 uninfected) for train image "
        "<STEM>.jpg — output verify_crops_<STEM>.jpg (or _academic.png with --academic)",
    )
    parser.add_argument(
        "--academic",
        action="store_true",
        help="With --crop-row: Parasitized/Uninfected labels + detector confidence; saves PNG",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="YOLO best.pt for confidence scores (default: first runs/detect/**/weights/best.pt)",
    )
    args = parser.parse_args()

    # Define where training images are stored
    train_imgs = DATA_DIR / "images" / "train"
    # Check if training images folder exists
    # If not, dataset hasn’t been prepared yet
    if not train_imgs.exists():
        print("Run create_splits.py and convert_to_yolo.py first.")
        return
    
    # Create folder to store verification results
    # project/data/verify_samples
    output_dir = PROJECT_ROOT / "data" / "verify_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_labels = DATA_DIR / "labels" / "train"

    if args.crop_row:
        stem = args.crop_row
        if stem.endswith(".jpg"):
            stem = Path(stem).stem
        img_path = train_imgs / f"{stem}.jpg"
        label_path = train_labels / f"{stem}.txt"
        if not img_path.is_file():
            print(f"Image not found: {img_path}")
            return
        if not label_path.is_file():
            print(f"Labels not found: {label_path}")
            return
        try:
            if args.academic:
                wpath = args.weights if args.weights is not None else _default_yolo_weights()
                if wpath is None or not wpath.is_file():
                    print(
                        "No --weights and no runs/detect/**/weights/best.pt; "
                        "confidence will show as —"
                    )
                out_path = output_dir / f"verify_crops_academic_{stem}.png"
                save_four_crop_row_academic(
                    img_path,
                    label_path,
                    out_path,
                    weights=wpath if wpath and wpath.is_file() else None,
                )
            else:
                out_path = output_dir / f"verify_crops_{stem}.jpg"
                save_four_crop_row(img_path, label_path, out_path)
        except ValueError as e:
            print(e)
        return

    if args.prefix:
        img_files = list(train_imgs.glob(f"{args.prefix}*.jpg"))
        if not img_files:
            print(f"No images found for {args.prefix}")
            return
    else:
        # Take first 3 images in training set
        img_files = list(train_imgs.glob("*.jpg"))[:3]
        # Add 247C99P60ThinF (has parasitized cells) so you can see red boxes
        extra = list(train_imgs.glob("247C99P60ThinF*.jpg"))
        img_files = list(dict.fromkeys(img_files + extra[:1]))  # avoid duplicates

    prefix_out = "verify_boxes_" if args.boxes_only else "verify_"
    
    # process each image
    for img_path in img_files:
        # Find corresponding label file
        label_path = train_labels / (img_path.stem + ".txt")
        # If label file exists, draw boxes and save to verify_samples folder
        if label_path.exists():
            out_path = output_dir / f"{prefix_out}{img_path.name}"
            draw_boxes(
                img_path,
                label_path,
                out_path,
                show_labels=not args.boxes_only,
            )

    if args.boxes_only:
        print(f"\nCheck {output_dir} — red=parasitized, green=uninfected (boxes only)")
    else:
        print(f"\nCheck {output_dir} — red=p (parasitized), green=u (uninfected)")


if __name__ == "__main__":
    main()
