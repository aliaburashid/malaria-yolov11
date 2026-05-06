"""
Step 4 (robustness reporting): load YOLO and two-stage metric CSVs, print comparison tables,
write a long-form drops summary CSV.

Inputs (same repository):
  - scripts/robustness/step2_run_yolo_robustness.py → YOLO_CSV
      Columns: condition, detection_f1, e2e_f1, cls_accuracy.
      Values derive from subprocess calls to scripts/two_stage_baseline/step4_evaluate_two_stage.py
      (greedy IoU matching), not Ultralytics model.val() mAP.
  - scripts/robustness/step3_run_two_stage_robustness.py → TWO_STAGE_CSV
      Same column layout as YOLO_CSV.

This script computes drop vs clean on end-to-end F1 only:
  drop(condition) = e2e_f1(condition) − e2e_f1(clean)  when condition != clean
  drop(clean) = 0.0

References:
  - csv — https://docs.python.org/3/library/csv.html
  - pathlib — https://docs.python.org/3/library/pathlib.html

Run from project root after steps 2 and 3:
  python3 scripts/robustness/step4_report_robustness.py
"""

# CSV read/write; DictReader yields dicts keyed by column names.
# Ref: https://docs.python.org/3/library/csv.html
import csv
# Paths anchored at repo root regardless of current working directory.
# Ref: https://docs.python.org/3/library/pathlib.html
from pathlib import Path
# Type hints on function signatures.
# Ref: https://docs.python.org/3/library/typing.html
from typing import List, Optional

# Repository root: scripts/robustness/ → three parents (matches step2_run_yolo_robustness.py repo_root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Artifact directory for robustness metrics (typically gitignored under runs/).
RUNS_ROBUSTNESS = PROJECT_ROOT / "runs" / "robustness"

# Default output path from scripts/robustness/step2_run_yolo_robustness.py (--out_csv).
# Headers: condition, detection_f1, e2e_f1, cls_accuracy (see DictWriter in that file).
YOLO_CSV = RUNS_ROBUSTNESS / "yolo_robustness_metrics.csv"

# Default output path from scripts/robustness/step3_run_two_stage_robustness.py (--out_csv).
TWO_STAGE_CSV = RUNS_ROBUSTNESS / "two_stage_robustness_metrics.csv"

# Long-form output: model, condition, metric, value, clean_value, drop — for tables and plotting.
OUT_SUMMARY = RUNS_ROBUSTNESS / "robustness_drops_summary.csv"


def load_csv(path: Path) -> List[dict]:
    """
    Load a CSV into a list of row dictionaries.

    Uses csv.DictReader after an existence check.
    Ref: https://docs.python.org/3/library/csv.html#csv.DictReader
    """
    # Missing file: empty list (caller treats as no data).
    if not path.exists():
        return []
    # newline="" recommended for csv module interoperability.
    with open(path, newline="") as f:
        # Values are strings until callers apply float().
        return list(csv.DictReader(f))


def get_clean_row(rows: List[dict]) -> Optional[dict]:
    """
    Return the row with condition == 'clean', or None.

    Baseline slug matches the clean folder name from the corruption dataset layout.
    """
    for r in rows:
        # .get tolerates rows missing the condition field.
        if r.get("condition") == "clean":
            return r
    return None


def main():
    """
    Load CSVs, print aligned metric tables, write OUT_SUMMARY.

    Legacy YOLO CSVs may use column "F1" instead of "e2e_f1"; readers below fall back accordingly.
    """
    yolo_rows = load_csv(YOLO_CSV)
    two_stage_rows = load_csv(TWO_STAGE_CSV)

    if not yolo_rows and not two_stage_rows:
        print("No metrics CSVs found.")
        print(f"  {YOLO_CSV}")
        print(f"  {TWO_STAGE_CSV}")
        return

    # Baselines for drop = value − clean_e2e.
    yolo_clean = get_clean_row(yolo_rows)
    two_stage_clean = get_clean_row(two_stage_rows)

    all_conditions = set()
    for r in yolo_rows:
        all_conditions.add(r["condition"])
    for r in two_stage_rows:
        all_conditions.add(r["condition"])
    # clean first, then alphabetical order for the remaining conditions.
    all_conditions = ["clean"] + sorted(c for c in all_conditions if c != "clean")

    summary_rows = []

    # YOLO metrics CSV (step2_run_yolo_robustness.py).
    if yolo_rows and yolo_clean:
        print("=" * 60)
        print("YOLO (greedy IoU + Step 4: detection F1, end-to-end F1, matched-label accuracy)")
        print("=" * 60)
        print(f"{'Condition':<25} {'Clean E2E':<12} {'E2E F1':<10} {'Drop':<10} {'Det F1':<10} {'Cls Acc':<10}")
        print("-" * 60)
        for cond in all_conditions:
            row = next((r for r in yolo_rows if r["condition"] == cond), None)
            if not row:
                continue
            # e2e_f1 from Step 4 parsing; "F1" names legacy step2 Ultralytics val() exports.
            end_to_end = float(row.get("e2e_f1", row.get("F1", 0)))
            detection = float(row.get("detection_f1", 0))
            matched_acc = float(row.get("cls_accuracy", 0))
            clean_end_to_end = float(yolo_clean.get("e2e_f1", yolo_clean.get("F1", 0)))
            drop = end_to_end - clean_end_to_end if cond != "clean" else 0.0
            print(
                f"{cond:<25} {clean_end_to_end:<12.4f} {end_to_end:<10.4f} {drop:+.4f}    "
                f"{detection:<10.4f} {matched_acc:<10.4f}"
            )
            summary_rows.append({
                "model": "yolo",
                "condition": cond,
                "metric": "e2e_f1",
                "value": end_to_end,
                "clean_value": clean_end_to_end,
                "drop": drop,
            })

    # Two-stage metrics CSV (step3_run_two_stage_robustness.py).
    if two_stage_rows and two_stage_clean:
        print()
        print("=" * 60)
        print("Two-stage (End-to-end F1, Detection F1, Cls Accuracy)")
        print("=" * 60)
        print(f"{'Condition':<25} {'Clean E2E F1':<12} {'E2E F1':<10} {'Drop':<10} {'Det F1':<10} {'Cls Acc':<10}")
        print("-" * 60)
        for cond in all_conditions:
            row = next((r for r in two_stage_rows if r["condition"] == cond), None)
            if not row:
                continue
            e2e = float(row.get("e2e_f1", 0))
            det_f1 = float(row.get("detection_f1", 0))
            acc = float(row.get("cls_accuracy", 0))
            clean_e2e = float(two_stage_clean.get("e2e_f1", 0))
            drop = e2e - clean_e2e if cond != "clean" else 0.0
            print(f"{cond:<25} {clean_e2e:<12.4f} {e2e:<10.4f} {drop:+.4f}    {det_f1:<10.4f} {acc:<10.4f}")
            summary_rows.append({
                "model": "two_stage",
                "condition": cond,
                "metric": "e2e_f1",
                "value": e2e,
                "clean_value": clean_e2e,
                "drop": drop,
            })

    RUNS_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "metric", "value", "clean_value", "drop"])
        w.writeheader()
        w.writerows(summary_rows)
    print()
    print(f"Wrote {OUT_SUMMARY}")


# Guard so main() runs only when the file is executed as a script.
# Ref: https://docs.python.org/3/library/__main__.html
if __name__ == "__main__":
    main()
