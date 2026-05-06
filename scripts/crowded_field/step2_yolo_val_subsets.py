"""
Step 2 — run YOLO validation separately on crowded and sparse test subsets.

Method:
- Create subset dataset YAML files whose `test` field points to split lists from step1.
- Run Ultralytics `model.val(..., split="test", conf=0.25, imgsz=640, batch=8)`.
- Save per-class and macro-F1 JSON artifacts for summary scripts.

References:
- scripts/crowded_field/README.md (protocol and constraints)
- Ultralytics val mode: https://docs.ultralytics.com/modes/val/

Run from project root (after step1):
  python3 scripts/crowded_field/step2_yolo_val_subsets.py
"""

from __future__ import annotations

# JSON export for subset metric artifacts.
import json
# Path handling independent of current working directory.
from pathlib import Path

# NumPy used for stable macro mean on class F1 array.
import numpy as np

# Repository root anchor.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Dataset and split-list inputs (step1 outputs the two txt files).
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
TXT_CROWDED = SPLITS_DIR / "test_crowded.txt"
TXT_SPARSE = SPLITS_DIR / "test_sparse.txt"
# Condition D detector checkpoint used across crowded-field evaluation.
WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "malaria_oversampled_weighted" / "weights" / "best.pt"
# Output root for YOLO subset validation runs.
EVAL_ROOT = PROJECT_ROOT / "runs" / "detect" / "crowded_field_eval"
YAML_DIR = EVAL_ROOT / "yaml"


def write_dataset_yaml(name: str, list_txt: Path) -> Path:
    # Create one dataset YAML per subset with custom test list.
    YAML_DIR.mkdir(parents=True, exist_ok=True)
    # Split lists use absolute image paths from step1.
    # Reference: scripts/crowded_field/README.md.
    yaml_path = YAML_DIR / f"dataset_{name}.yaml"
    abs_root = DATA_PROCESSED.resolve()
    abs_list = list_txt.resolve()
    text = (
        f"path: {abs_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: {abs_list}\n"
        "names:\n"
        "  0: parasitized\n"
        "  1: uninfected\n"
        "nc: 2\n"
    )
    yaml_path.write_text(text)
    return yaml_path


def run_val(weights: Path, data_yaml: Path, project: Path, run_name: str) -> object:
    # Ultralytics model API used for subset evaluation.
    # Reference: https://docs.ultralytics.com/modes/val/
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # Params aligned with crowded-field protocol (conf=0.25, imgsz=640, batch=8).
    return model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        conf=0.25,
        batch=8,
        project=str(project),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )


def save_subset_metrics(metrics, out_dir: Path) -> None:
    # Persist compact machine-readable metrics for step4_summary.py fallback path.
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_arr = np.array(metrics.box.f1, dtype=np.float64)
    macro = float(np.mean(f1_arr))
    per_class = [float(x) for x in f1_arr]
    (out_dir / "subset_val_metrics.json").write_text(
        json.dumps({"f1_per_class": per_class, "f1_macro": macro}, indent=2) + "\n"
    )


def print_per_class(metrics, label: str) -> None:
    # Print per-class box metrics reported by Ultralytics validation outputs.
    class_names = ["parasitized", "uninfected"]
    print(f"\n{label} — per-class P/R/F1 (Ultralytics box metrics)")
    for i, cname in enumerate(class_names):
        p, r, m50, m5095 = metrics.box.class_result(i)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  {cname:12} P={p:.4f} R={r:.4f} F1={f1:.4f}  mAP50={m50:.4f} mAP50-95={m5095:.4f}")
    overall = metrics.box.mean_results()
    print(
        f"  {'overall':12} P={overall[0]:.4f} R={overall[1]:.4f}  "
        f"mAP50={overall[2]:.4f} mAP50-95={overall[3]:.4f}"
    )


def main() -> None:
    # Validate required inputs from training (weights) and step1 (split lists).
    if not WEIGHTS.exists():
        raise SystemExit(f"Missing weights: {WEIGHTS}")
    if not TXT_CROWDED.exists() or not TXT_SPARSE.exists():
        raise SystemExit(f"Run step1 first. Missing {TXT_CROWDED} or {TXT_SPARSE}")

    # Build subset-specific dataset YAML files.
    yaml_c = write_dataset_yaml("crowded", TXT_CROWDED)
    yaml_s = write_dataset_yaml("sparse", TXT_SPARSE)

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("Crowded-field Step 2 — YOLO val on subsets (conf=0.25)")
    # Run crowded subset, save json artifact, print class table.
    m_c = run_val(WEIGHTS, yaml_c, EVAL_ROOT, "yolo_crowded")
    save_subset_metrics(m_c, EVAL_ROOT / "yolo_crowded")
    print_per_class(m_c, "Crowded subset")
    print(f"\nSaved run folder: {EVAL_ROOT / 'yolo_crowded'}")

    # Run sparse subset, save json artifact, print class table.
    m_s = run_val(WEIGHTS, yaml_s, EVAL_ROOT, "yolo_sparse")
    save_subset_metrics(m_s, EVAL_ROOT / "yolo_sparse")
    print_per_class(m_s, "Sparse subset")
    print(f"\nSaved run folder: {EVAL_ROOT / 'yolo_sparse'}")
    print("Next: python3 scripts/crowded_field/step3_two_stage_subset_metrics.py")


if __name__ == "__main__":
    main()
