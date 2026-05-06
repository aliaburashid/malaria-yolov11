"""
Step 3 — compute two-stage metrics on crowded and sparse test subsets.

Method:
- Filter runs/two_stage_baseline/predictions_test_finetuned.json by image stems
  listed in data/splits/test_crowded.txt and data/splits/test_sparse.txt.
- Reuse greedy one-to-one IoU matching from scripts/two_stage_baseline/step4_evaluate_two_stage.py.
- Report detection metrics, end-to-end metrics, matched classification accuracy,
  and per-class end-to-end P/R/F1 plus macro F1.

References:
- scripts/crowded_field/README.md
- scripts/two_stage_baseline/step4_evaluate_two_stage.py

Run from project root (after step1):
  python3 scripts/crowded_field/step3_two_stage_subset_metrics.py
"""

from __future__ import annotations

# Dynamic import for reusing baseline evaluator helpers.
import importlib.util
# JSON predictions input from two-stage inference.
import json
# stderr/exit handling.
import sys
from pathlib import Path
from typing import Dict, Set

# Image size is needed to denormalize YOLO labels in step4 loader.
from PIL import Image

# Repository root + crowded-field paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_PATH = PROJECT_ROOT / "runs" / "two_stage_baseline" / "predictions_test_finetuned.json"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TXT_CROWDED = SPLITS_DIR / "test_crowded.txt"
TXT_SPARSE = SPLITS_DIR / "test_sparse.txt"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
LABELS_DIR = DATA_ROOT / "labels" / "test"
EVAL_ROOT = PROJECT_ROOT / "runs" / "detect" / "crowded_field_eval"
OUT_CROWDED = EVAL_ROOT / "two_stage_crowded_results.txt"
OUT_SPARSE = EVAL_ROOT / "two_stage_sparse_results.txt"


def load_step4_module():
    # Import canonical matching/eval logic without editing baseline file.
    # Reference: scripts/two_stage_baseline/step4_evaluate_two_stage.py
    path = PROJECT_ROOT / "scripts" / "two_stage_baseline" / "step4_evaluate_two_stage.py"
    spec = importlib.util.spec_from_file_location("step4_evaluate_two_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stems_from_list_file(list_path: Path) -> Set[str]:
    # Convert absolute list-file paths from step1 into stem set for filtering.
    stems: Set[str] = set()
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stems.add(Path(line).stem)
    return stems


def filter_images(images: Dict, allowed_stems: Set[str]) -> Dict:
    # Keep prediction entries only for the requested crowded/sparse subset.
    out = {}
    for rel_key, entry in images.items():
        stem = Path(rel_key).stem
        if stem in allowed_stems:
            out[rel_key] = entry
    return out


def _accumulate_per_class_tp_fp_fn(
    matches: list,
    dets: list,
    gt_boxes: list,
    tp: list,
    fp: list,
    fn: list,
) -> None:
    """
    Per-class counts aligned with end-to-end semantics: TP for class c only if
    greedy IoU match exists and pred.cls == gt.cls == c. Same matches as global
    detection (IoU >= threshold, one-to-one).
    """
    pred_to_gt = {pi: gi for pi, gi in matches}
    gt_to_pred = {gi: pi for pi, gi in matches}

    for pi, gi in matches:
        pc = dets[pi]["cls"]
        gc = gt_boxes[gi][0]
        if pc == gc == 0:
            tp[0] += 1
        elif pc == gc == 1:
            tp[1] += 1

    for pi, det in enumerate(dets):
        c = det["cls"]
        if c not in (0, 1):
            continue
        if pi not in pred_to_gt:
            fp[c] += 1
        else:
            gi = pred_to_gt[pi]
            if gt_boxes[gi][0] != c:
                fp[c] += 1

    for gi, (gc, _) in enumerate(gt_boxes):
        if gc not in (0, 1):
            continue
        if gi not in gt_to_pred:
            fn[gc] += 1
        else:
            pi = gt_to_pred[gi]
            if dets[pi]["cls"] != gc:
                fn[gc] += 1


def evaluate_subset(
    mod,
    images_data: Dict,
    subset_name: str,
    iou: float,
):
    """
    Returns detection (p,r,f1), e2e (p,r,f1), cls_accuracy, n_images,
    then per-class (parasitized=0, uninfected=1) P/R/F1 and macro mean F1.
    """
    # Global detection counters under class-agnostic greedy matching.
    total_tp = total_fp = total_fn = 0
    # Matched-pair class counters for end-to-end/classification metrics.
    total_matched = total_cls_correct = 0
    # Per-class TP/FP/FN for parasitized(0) and uninfected(1).
    tp_c = [0, 0]
    fp_c = [0, 0]
    fn_c = [0, 0]
    n_images = 0
    skipped_no_image = 0
    skipped_no_label = 0

    # Evaluate one image entry at a time from predictions JSON.
    for rel_key, img_entry in images_data.items():
        dets = img_entry.get("dets", [])
        img_path = PROJECT_ROOT / rel_key
        if not img_path.exists():
            stem = Path(rel_key).stem
            img_path = DATA_ROOT / "images" / "test" / (stem + ".jpg")
            if not img_path.exists():
                img_path = DATA_ROOT / "images" / "test" / (stem + ".png")

        if not img_path.exists():
            skipped_no_image += 1
            continue

        n_images += 1
        try:
            w, h = Image.open(img_path).size
        except Exception:
            skipped_no_image += 1
            continue

        stem = img_path.stem
        label_path = LABELS_DIR / (stem + ".txt")
        if not label_path.exists():
            skipped_no_label += 1
            continue

        # Reuse baseline GT loader (normalized YOLO labels -> pixel xyxy).
        gt_boxes = mod.load_gt_boxes(label_path, w, h)

        if not gt_boxes and not dets:
            continue
        if not gt_boxes:
            total_fp += len(dets)
            for det in dets:
                c = det.get("cls")
                if c in (0, 1):
                    fp_c[c] += 1
            continue
        if not dets:
            total_fn += len(gt_boxes)
            for gc, _ in gt_boxes:
                if gc in (0, 1):
                    fn_c[gc] += 1
            continue

        pred_xyxy = [d["xyxy"] for d in dets]
        # Reuse baseline greedy one-to-one IoU matching implementation.
        matches = mod.match_predictions_to_gt(pred_xyxy, gt_boxes, iou)
        tp = len(matches)
        total_tp += tp
        total_fp += len(dets) - tp
        total_fn += len(gt_boxes) - tp
        total_matched += tp
        for pi, gi in matches:
            if dets[pi]["cls"] == gt_boxes[gi][0]:
                total_cls_correct += 1

        _accumulate_per_class_tp_fp_fn(matches, dets, gt_boxes, tp_c, fp_c, fn_c)

    total_preds = total_tp + total_fp
    total_gt = total_tp + total_fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    cls_accuracy = total_cls_correct / total_matched if total_matched > 0 else 0.0

    tp_e2e = total_cls_correct
    fp_e2e = total_preds - tp_e2e
    fn_e2e = total_gt - tp_e2e
    prec_e2e = tp_e2e / (tp_e2e + fp_e2e) if (tp_e2e + fp_e2e) > 0 else 0.0
    rec_e2e = tp_e2e / (tp_e2e + fn_e2e) if (tp_e2e + fn_e2e) > 0 else 0.0
    f1_e2e = (
        2 * prec_e2e * rec_e2e / (prec_e2e + rec_e2e) if (prec_e2e + rec_e2e) > 0 else 0.0
    )

    if skipped_no_image:
        print(f"  [{subset_name}] Warning: skipped {skipped_no_image} images (not found).")
    if skipped_no_label:
        print(f"  [{subset_name}] Warning: skipped {skipped_no_label} images (no label).")

    # Local helper to convert TP/FP/FN counts to precision/recall/F1.
    def prf1(tc: int, fpc: int, fnc: int) -> tuple[float, float, float]:
        p = tc / (tc + fpc) if (tc + fpc) > 0 else 0.0
        r = tc / (tc + fnc) if (tc + fnc) > 0 else 0.0
        f1v = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1v

    p0, r0, f1_0 = prf1(tp_c[0], fp_c[0], fn_c[0])
    p1, r1, f1_1 = prf1(tp_c[1], fp_c[1], fn_c[1])
    macro_f1 = (f1_0 + f1_1) / 2.0

    return (
        precision,
        recall,
        f1,
        prec_e2e,
        rec_e2e,
        f1_e2e,
        cls_accuracy,
        n_images,
        p0,
        r0,
        f1_0,
        p1,
        r1,
        f1_1,
        macro_f1,
    )


def format_report(
    subset: str,
    det_p: float,
    det_r: float,
    det_f1: float,
    e2e_p: float,
    e2e_r: float,
    e2e_f1: float,
    cls_acc: float,
    n_img: int,
    p0: float,
    r0: float,
    f1_0: float,
    p1: float,
    r1: float,
    f1_1: float,
    macro_f1: float,
) -> str:
    # Keep key names stable: step4_summary.parse_two_stage_results() parses them.
    lines = [
        f"subset: {subset}",
        f"images_evaluated: {n_img}",
        f"detection_precision: {det_p:.6f}",
        f"detection_recall: {det_r:.6f}",
        f"detection_f1: {det_f1:.6f}",
        f"end_to_end_precision: {e2e_p:.6f}",
        f"end_to_end_recall: {e2e_r:.6f}",
        f"end_to_end_f1: {e2e_f1:.6f}",
        f"classification_accuracy_matched: {cls_acc:.6f}",
        "",
        "# Per-class end-to-end (TP = IoU match AND pred class == GT class for that class)",
        f"per_class_parasitized_precision: {p0:.6f}",
        f"per_class_parasitized_recall: {r0:.6f}",
        f"per_class_parasitized_f1: {f1_0:.6f}",
        f"per_class_uninfected_precision: {p1:.6f}",
        f"per_class_uninfected_recall: {r1:.6f}",
        f"per_class_uninfected_f1: {f1_1:.6f}",
        f"per_class_macro_f1: {macro_f1:.6f}",
        "",
        "Summary:",
        f"  Detection  P={det_p:.4f} R={det_r:.4f} F1={det_f1:.4f}",
        f"  End-to-end P={e2e_p:.4f} R={e2e_r:.4f} F1={e2e_f1:.4f}",
        f"  Classification accuracy (on matched): {cls_acc:.4f}",
        f"  Per-class (e2e TP rule): parasitized F1={f1_0:.4f}, uninfected F1={f1_1:.4f}, macro F1={macro_f1:.4f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not PRED_PATH.exists():
        print(f"ERROR: Missing predictions: {PRED_PATH}", file=sys.stderr)
        sys.exit(1)
    if not TXT_CROWDED.exists() or not TXT_SPARSE.exists():
        print("ERROR: Run step1 first (split list files missing).", file=sys.stderr)
        sys.exit(1)

    mod = load_step4_module()
    iou = float(mod.IOU_THRESH)

    with open(PRED_PATH) as f:
        data = json.load(f)
    images_all = data.get("images", {})
    if not images_all:
        print("ERROR: No images in predictions JSON", file=sys.stderr)
        sys.exit(1)

    stems_c = stems_from_list_file(TXT_CROWDED)
    stems_s = stems_from_list_file(TXT_SPARSE)
    img_c = filter_images(images_all, stems_c)
    img_s = filter_images(images_all, stems_s)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("Crowded-field Step 3 — two-stage subset metrics (greedy IoU >= 0.5)")
    print(f"  Predictions: {PRED_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Crowded images in JSON: {len(img_c)} (stems in list: {len(stems_c)})")
    print(f"  Sparse images in JSON:   {len(img_s)} (stems in list: {len(stems_s)})")

    res_c = evaluate_subset(mod, img_c, "crowded", iou)
    res_s = evaluate_subset(mod, img_s, "sparse", iou)
    (
        dpc,
        drc,
        dfc,
        epc,
        erc,
        efc,
        cac,
        nic,
        p0c,
        r0c,
        f10c,
        p1c,
        r1c,
        f11c,
        mfc,
    ) = res_c
    (
        dps,
        drs,
        dfs,
        eps,
        ers,
        efs,
        cas,
        nis,
        p0s,
        r0s,
        f10s,
        p1s,
        r1s,
        f11s,
        mfs,
    ) = res_s

    txt_c = format_report(
        "crowded",
        dpc,
        drc,
        dfc,
        epc,
        erc,
        efc,
        cac,
        nic,
        p0c,
        r0c,
        f10c,
        p1c,
        r1c,
        f11c,
        mfc,
    )
    txt_s = format_report(
        "sparse",
        dps,
        drs,
        dfs,
        eps,
        ers,
        efs,
        cas,
        nis,
        p0s,
        r0s,
        f10s,
        p1s,
        r1s,
        f11s,
        mfs,
    )
    OUT_CROWDED.write_text(txt_c)
    OUT_SPARSE.write_text(txt_s)

    print("\nCrowded subset:")
    print(txt_c)
    print("Sparse subset:")
    print(txt_s)
    print(f"Wrote {OUT_CROWDED.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUT_SPARSE.relative_to(PROJECT_ROOT)}")
    print("Next: python3 scripts/crowded_field/step4_summary.py")


if __name__ == "__main__":
    main()
