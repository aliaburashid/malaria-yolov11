"""
Evaluate Conditions A/B/C/D and print comparison table.
A: baseline (no weighting, normal sampling)
B: weighted loss
C: oversampling (no weighting)
D: oversampling + weighted loss

Run from project root: python3 scripts/evaluate_conditions.py
Source: Ultralytics val API and metrics (https://github.com/ultralytics/ultralytics)
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "config" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
# (label, run_name subdir)
CONDITIONS = [
    ("Baseline (A)", "malaria"),
    ("Weighted (B)", "malaria_weighted"),
    ("Oversampled (C)", "malaria_oversampled"),
    ("Oversampled+Weighted (D)", "malaria_oversampled_weighted"),
]
CLASS_NAMES = ["parasitized", "uninfected"]


def run_val(weights_path: Path, split: str = "val"):
    """Run validation and return per-class (P, R, mAP50, mAP50-95) and overall metrics."""
    from ultralytics import YOLO

    if not weights_path.exists():
        return None
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(DATASET_PATH), split=split, imgsz=640, batch=8, verbose=False)
    # metrics is DetMetrics; metrics.box has class_result(i) -> (precision, recall, mAP50, mAP50-95)
    nc = len(CLASS_NAMES)
    per_class = []
    for i in range(nc):
        p, r, m50, m5095 = metrics.box.class_result(i)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class.append({"name": CLASS_NAMES[i], "P": p, "R": r, "F1": f1, "mAP50": m50, "mAP50-95": m5095})
    overall = metrics.box.mean_results()  # [P, R, mAP50, mAP50-95]
    return {"per_class": per_class, "overall_P": overall[0], "overall_R": overall[1], "overall_mAP50": overall[2], "overall_mAP50-95": overall[3]}


def table_from_csv():
    """Build comparison table from results.csv (overall metrics only) when val not run."""
    baseline_csv = PROJECT_ROOT / "runs" / "detect" / "malaria" / "results.csv"
    weighted_csv = PROJECT_ROOT / "runs" / "detect" / "malaria_weighted" / "results.csv"
    out = []
    for label, path in [("Baseline (A)", baseline_csv), ("Weighted (B)", weighted_csv)]:
        if not path.exists():
            out.append((label, None))
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            out.append((label, None))
            continue
        # Use last row (final epoch) and also find best mAP50 row
        last = rows[-1]
        best_row = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
        out.append((label, {
            "last": last,
            "best_mAP50": best_row,
            "best_epoch": int(best_row["epoch"]),
        }))
    return out


def main():
    print("=" * 70)
    print("Conditions A/B/C/D — Comparison (Val set)")
    print("=" * 70)

    # 1) Run validation for each condition that has best.pt
    results = {}
    for label, run_name in CONDITIONS:
        wpath = RUNS_DIR / run_name / "weights" / "best.pt"
        if wpath.exists():
            results[(label, run_name)] = run_val(wpath)
        else:
            results[(label, run_name)] = None

    # 2) Per-class metrics
    FMT = ".2f"  # two decimal places consistently (academic standard)
    any_res = any(r is not None for r in results.values())
    if any_res:
        print("\n--- Per-class metrics (from validation, best.pt) ---\n")
        for (label, run_name), res in results.items():
            if res is None:
                print(f"{label}: (no best.pt)\n")
                continue
            print(f"{label}")
            for c in res["per_class"]:
                print(f"  {c['name']:12}  P={c['P']:{FMT}}  R={c['R']:{FMT}}  F1={c['F1']:{FMT}}  mAP50={c['mAP50']:{FMT}}  mAP50-95={c['mAP50-95']:{FMT}}")
            print(f"  {'overall':12}  P={res['overall_P']:{FMT}}  R={res['overall_R']:{FMT}}  mAP50={res['overall_mAP50']:{FMT}}  mAP50-95={res['overall_mAP50-95']:{FMT}}\n")

    # 3) Dissertation table (2 dp, Parasitized R/P/F1, Uninfected R, mAP50, mAP50-95)
    print("\n--- Dissertation table (Val set) ---\n")
    rows = [(label, res) for (label, _), res in results.items() if res is not None]
    if rows:
        print(f"{'Condition':<28} {'Parasitized R':<12} {'Parasitized P':<12} {'Parasitized F1':<14} {'Uninfected R':<12} {'mAP50':<8} {'mAP50-95':<10}")
        print("-" * 100)
        for label, res in rows:
            p0, p1 = res["per_class"][0], res["per_class"][1]
            print(f"{label:<28} {p0['R']:<12.2f} {p0['P']:<12.2f} {p0['F1']:<14.2f} {p1['R']:<12.2f} {res['overall_mAP50']:<8.2f} {res['overall_mAP50-95']:<10.2f}")
    else:
        print("No best.pt found. Train A/B/C/D then re-run.")

    # 4) Save table to CSV
    out_csv = RUNS_DIR / "condition_comparison.csv"
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Model", "Parasitized_R", "Parasitized_P", "Parasitized_F1", "Uninfected_R", "Uninfected_P", "mAP50", "mAP50-95"])
            for label, res in rows:
                p0, p1 = res["per_class"][0], res["per_class"][1]
                slug = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
                w.writerow([slug, round(p0["R"], 2), round(p0["P"], 2), round(p0["F1"], 2), round(p1["R"], 2), round(p1["P"], 2), round(res["overall_mAP50"], 2), round(res["overall_mAP50-95"], 2)])
        print(f"\nTable saved to: {out_csv}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
