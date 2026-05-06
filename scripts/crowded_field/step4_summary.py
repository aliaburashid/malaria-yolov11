"""
Step 4 — build a crowded-vs-sparse summary table across pipelines.

Preferred inputs (greedy IoU protocol):
  runs/detect/crowded_field_eval/yolo_crowded_greedy_results.json
  runs/detect/crowded_field_eval/yolo_sparse_greedy_results.json
  runs/detect/crowded_field_eval/two_stage_{crowded,sparse}_results.txt

Fallback inputs (legacy YOLO val macro F1 only):
  runs/detect/crowded_field_eval/yolo_{crowded,sparse}/subset_val_metrics.json

Output:
  runs/detect/crowded_field_eval/crowded_field_summary.csv

Definition:
  delta_crowded_minus_sparse = f1_crowded - f1_sparse

Run order:
  python3 scripts/crowded_field/step2b_yolo_subset_greedy_metrics.py
  python3 scripts/crowded_field/step4_summary.py
"""

from __future__ import annotations

# CSV output table writer.
import csv
# JSON loader for YOLO greedy / fallback artifacts.
import json
# stderr and non-zero exits for missing inputs.
import sys
from pathlib import Path

# Repository root and crowded-field evaluation artifact paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / "runs" / "detect" / "crowded_field_eval"
OUT_CSV = EVAL_ROOT / "crowded_field_summary.csv"


def load_yolo_greedy_metrics() -> dict[str, tuple[float, float, float]]:
    """Return (crowded, sparse) tuples for (e2e_f1, det_f1, per_class_macro_f1) from Step 2b JSON."""
    # Preferred inputs from step2b greedy protocol.
    paths = {
        "crowded": EVAL_ROOT / "yolo_crowded_greedy_results.json",
        "sparse": EVAL_ROOT / "yolo_sparse_greedy_results.json",
    }
    out: dict[str, tuple[float, float, float]] = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — run: python3 scripts/crowded_field/step2b_yolo_subset_greedy_metrics.py")
        with open(p) as f:
            d = json.load(f)
        out[name] = (
            # Keys are emitted by step2b to_json_report().
            float(d["end_to_end_f1"]),
            float(d["detection_f1"]),
            float(d["per_class_macro_f1"]),
        )
    e2e_c, det_c, macro_c = out["crowded"]
    e2e_s, det_s, macro_s = out["sparse"]
    return {
        "e2e": (e2e_c, e2e_s),
        "det": (det_c, det_s),
        "macro": (macro_c, macro_s),
    }


def load_yolo_val_macro_fallback() -> tuple[float, float]:
    """Legacy: Ultralytics val box F1 macro (not greedy E2E)."""
    # Fallback keeps script usable if step2b was not run.
    # Reference: scripts/crowded_field/step2_yolo_val_subsets.py -> subset_val_metrics.json
    print(
        "WARNING: greedy YOLO JSONs not found; falling back to Ultralytics val f1_macro "
        "(not comparable to thesis greedy E2E table). Run step2b_yolo_subset_greedy_metrics.py.",
        file=sys.stderr,
    )
    y_c = y_s = 0.0
    for subdir, acc in [("yolo_crowded", "crowded"), ("yolo_sparse", "sparse")]:
        p = EVAL_ROOT / subdir / "subset_val_metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — run step2_yolo_val_subsets.py or step2b.")
        with open(p) as f:
            d = json.load(f)
        v = float(d["f1_macro"])
        if acc == "crowded":
            y_c = v
        else:
            y_s = v
    return y_c, y_s


def parse_two_stage_results(path: Path) -> tuple[float, float, float]:
    # Parse key-value text report written by step3 format_report().
    # Reference: scripts/crowded_field/step3_two_stage_subset_metrics.py
    det_f1 = e2e_f1 = macro_f1 = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("detection_f1:"):
                det_f1 = float(line.split(":", 1)[1].strip())
            elif line.startswith("end_to_end_f1:"):
                e2e_f1 = float(line.split(":", 1)[1].strip())
            elif line.startswith("per_class_macro_f1:"):
                macro_f1 = float(line.split(":", 1)[1].strip())
    if det_f1 is None or e2e_f1 is None or macro_f1 is None:
        raise ValueError(
            f"Could not parse detection_f1 / end_to_end_f1 / per_class_macro_f1 from {path}"
        )
    return det_f1, e2e_f1, macro_f1


def main() -> None:
    # Prefer greedy YOLO artifacts; fallback to val-macro metrics when absent.
    greedy_c = EVAL_ROOT / "yolo_crowded_greedy_results.json"
    greedy_s = EVAL_ROOT / "yolo_sparse_greedy_results.json"
    yolo_greedy_extra: tuple[tuple[float, float], tuple[float, float]] | None
    if greedy_c.exists() and greedy_s.exists():
        g = load_yolo_greedy_metrics()
        y_e2e_c, y_e2e_s = g["e2e"]
        yolo_greedy_extra = (g["det"], g["macro"])
    else:
        y_e2e_c, y_e2e_s = load_yolo_val_macro_fallback()
        yolo_greedy_extra = None

    # Two-stage subset reports are mandatory for this summary.
    tc = EVAL_ROOT / "two_stage_crowded_results.txt"
    ts = EVAL_ROOT / "two_stage_sparse_results.txt"
    if not tc.exists() or not ts.exists():
        print(f"ERROR: Missing {tc} or {ts} — run step3 first.", file=sys.stderr)
        sys.exit(1)

    # Parse two-stage detection/e2e/macro metrics for both subsets.
    d_c_det, d_c_e2e, d_c_macro = parse_two_stage_results(tc)
    d_s_det, d_s_e2e, d_s_macro = parse_two_stage_results(ts)

    # Compose output rows depending on whether greedy YOLO metrics are available.
    if yolo_greedy_extra is not None:
        (y_det_c, y_det_s), (y_macro_c, y_macro_s) = yolo_greedy_extra[0], yolo_greedy_extra[1]
        rows: list[dict[str, float | str]] = [
            {
                "pipeline": "yolo_condition_d",
                "metric": "end_to_end_f1_greedy_iou0.5",
                "f1_crowded": y_e2e_c,
                "f1_sparse": y_e2e_s,
                "delta_crowded_minus_sparse": y_e2e_c - y_e2e_s,
            },
            {
                "pipeline": "two_stage_finetuned",
                "metric": "end_to_end_f1_greedy_iou0.5",
                "f1_crowded": d_c_e2e,
                "f1_sparse": d_s_e2e,
                "delta_crowded_minus_sparse": d_c_e2e - d_s_e2e,
            },
            {
                "pipeline": "yolo_condition_d",
                "metric": "detection_f1_greedy_iou0.5",
                "f1_crowded": y_det_c,
                "f1_sparse": y_det_s,
                "delta_crowded_minus_sparse": y_det_c - y_det_s,
            },
            {
                "pipeline": "yolo_condition_d",
                "metric": "per_class_macro_f1_greedy_iou0.5",
                "f1_crowded": y_macro_c,
                "f1_sparse": y_macro_s,
                "delta_crowded_minus_sparse": y_macro_c - y_macro_s,
            },
            {
                "pipeline": "two_stage_finetuned",
                "metric": "per_class_macro_f1_greedy_iou0.5",
                "f1_crowded": d_c_macro,
                "f1_sparse": d_s_macro,
                "delta_crowded_minus_sparse": d_c_macro - d_s_macro,
            },
            {
                "pipeline": "two_stage_finetuned",
                "metric": "detection_f1_greedy_iou0.5",
                "f1_crowded": d_c_det,
                "f1_sparse": d_s_det,
                "delta_crowded_minus_sparse": d_c_det - d_s_det,
            },
        ]
    else:
        rows = [
            {
                "pipeline": "yolo_condition_d",
                "metric": "box_f1_macro_ultralytics_val",
                "f1_crowded": y_e2e_c,
                "f1_sparse": y_e2e_s,
                "delta_crowded_minus_sparse": y_e2e_c - y_e2e_s,
            },
            {
                "pipeline": "two_stage_finetuned",
                "metric": "end_to_end_f1_greedy_iou0.5",
                "f1_crowded": d_c_e2e,
                "f1_sparse": d_s_e2e,
                "delta_crowded_minus_sparse": d_c_e2e - d_s_e2e,
            },
            {
                "pipeline": "two_stage_finetuned",
                "metric": "per_class_macro_f1_greedy_iou0.5",
                "f1_crowded": d_c_macro,
                "f1_sparse": d_s_macro,
                "delta_crowded_minus_sparse": d_c_macro - d_s_macro,
            },
            {
                "pipeline": "two_stage_finetuned",
                "metric": "detection_f1_greedy_iou0.5",
                "f1_crowded": d_c_det,
                "f1_sparse": d_s_det,
                "delta_crowded_minus_sparse": d_c_det - d_s_det,
            },
        ]

    # Write one flat CSV used in chapter tables/plots.
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pipeline",
                "metric",
                "f1_crowded",
                "f1_sparse",
                "delta_crowded_minus_sparse",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Crowded-field Step 4 — summary (delta = crowded − sparse)\n")
    print(
        f"{'Pipeline':<22} {'Metric':<42} {'F1 crow':>10} {'F1 spar':>10} {'Delta':>10}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r['pipeline']:<22} {r['metric']:<42} "
            f"{r['f1_crowded']:>10.4f} {r['f1_sparse']:>10.4f} {r['delta_crowded_minus_sparse']:>+10.4f}"
        )
    print(f"\nWrote {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
