"""
Publication-quality YOLOv11 inference: configurable conf/iou, labels without confidence, optional ROI zoom.
Run from project root: python3 scripts/run_publication_predictions.py
Or from notebook: %run scripts/run_publication_predictions.py
"""
from pathlib import Path
import numpy as np

# ---------- Publication-quality visualisation (configurable) ----------
CONF_THRES = 0.90
IOU_THRES = 0.45
SHOW_LABELS = True
SHOW_CONF = False
LINE_WIDTH = 2
MAX_DETS = 300
ROI = None  # Optional: (x1, y1, x2, y2) to save a cropped *_zoom.jpg; set to e.g. (100, 100, 500, 400) to enable
# ----------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    best_pt = root / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
    test_dir = root / "data" / "processed" / "images" / "test"
    out_dir = root / "runs" / "detect" / "summary_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not best_pt.exists():
        print("Weights not found:", best_pt)
        return

    from ultralytics import YOLO
    import cv2

    model = YOLO(str(best_pt))
    imgs = sorted(test_dir.glob("*.jpg"))[:3]
    if not imgs:
        print("No test images in", test_dir)
        return

    results = model.predict(
        source=[str(p) for p in imgs],
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DETS,
        save=False,
        verbose=True,
    )

    saved_paths = []
    for result in results:
        src_path = Path(result.path)
        out_name = src_path.name
        out_path = out_dir / out_name
        try:
            img = result.plot(conf=SHOW_CONF, labels=SHOW_LABELS, line_width=LINE_WIDTH)
        except TypeError:
            # Fallback: draw with Annotator (class name only, no confidence), respect CONF_THRES
            from ultralytics.utils.plotting import Annotator, colors
            orig = result.orig_img if hasattr(result, "orig_img") else np.array(result.plot())
            if isinstance(orig, np.ndarray):
                img = np.ascontiguousarray(orig.copy())
            else:
                img = np.ascontiguousarray(np.array(orig))
            if result.boxes is not None:
                annotator = Annotator(img, line_width=LINE_WIDTH)
                for d in result.boxes:
                    if float(d.conf) < CONF_THRES:
                        continue
                    c = int(d.cls)
                    label = (result.names[c] if SHOW_LABELS else "")
                    box = d.xyxy.squeeze()
                    bx = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
                    annotator.box_label(bx, label, color=colors(c, True))
                img = annotator.result()
            else:
                img = result.plot(line_width=LINE_WIDTH)
        if img is not None:
            cv2.imwrite(str(out_path), img)
            saved_paths.append(out_path)
            if ROI is not None:
                x1, y1, x2, y2 = ROI
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img[y1:y2, x1:x2]
                zoom_path = out_dir / (src_path.stem + "_zoom.jpg")
                cv2.imwrite(str(zoom_path), crop)
                saved_paths.append(zoom_path)

    print("Saved to:", out_dir)
    return out_dir, saved_paths


if __name__ == "__main__":
    main()
