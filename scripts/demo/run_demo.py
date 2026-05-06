"""
Demo entrypoint for running evaluation against a YOLO checkpoint (best.pt).

Actions:
- Ultralytics validation on config/dataset.yaml test split (data/processed).
- Optional robustness Step 2 run using scripts/robustness/step2_run_yolo_robustness.py.

References:
- Ultralytics val: https://docs.ultralytics.com/modes/val/
- Repository dataset YAML: config/dataset.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_YAML = PROJECT_ROOT / "config" / "dataset.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "demo"


def run_ultralytics_val(weights: Path, data_yaml: Path) -> dict:
    # Ultralytics YOLO API.
    # Ref: https://docs.ultralytics.com/modes/val/
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        conf=0.25,
        batch=8,
        verbose=False,
        plots=False,
    )
    overall = metrics.box.mean_results()  # [P, R, mAP50, mAP50-95]
    p, r, map50, map5095 = (float(overall[0]), float(overall[1]), float(overall[2]), float(overall[3]))
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "map50": map50, "map50_95": map5095}


def run_robustness_step2(weights: Path) -> int:
    step2 = PROJECT_ROOT / "scripts" / "robustness" / "step2_run_yolo_robustness.py"
    if not step2.exists():
        print(f"Missing robustness Step 2 script: {step2}", file=sys.stderr)
        return 2
    cmd = [
        sys.executable,
        str(step2),
        "--detector_weights",
        str(weights),
    ]
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


def iter_weight_files(
    weights: list[Path] | None,
    weights_dir: Path | None,
    pattern: str,
) -> list[Path]:
    # Combine explicit weights + directory scan into a single ordered list.
    out: list[Path] = []
    if weights:
        out.extend(weights)
    if weights_dir is not None:
        if not weights_dir.exists():
            raise FileNotFoundError(f"Missing weights_dir: {weights_dir}")
        if not weights_dir.is_dir():
            raise NotADirectoryError(f"weights_dir is not a directory: {weights_dir}")
        out.extend(sorted(weights_dir.rglob(pattern)))
    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def write_summary_csv(out_csv: Path, rows: Iterable[dict]) -> None:
    fieldnames = ["weights", "precision", "recall", "f1", "map50", "map50_95"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run demo evaluation for one or more YOLO checkpoints.")
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="*",
        default=None,
        help="One or more YOLO checkpoints (best.pt). If omitted, use --weights_dir scan.",
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        default=None,
        help="Directory to scan recursively for checkpoints (default: disabled).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="best.pt",
        help="Filename pattern for --weights_dir scan (default: best.pt).",
    )
    parser.add_argument(
        "--data_yaml",
        type=Path,
        default=DEFAULT_DATASET_YAML,
        help="Dataset YAML for Ultralytics val (default: config/dataset.yaml)",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument(
        "--run_robustness",
        action="store_true",
        help="Also run scripts/robustness/step2_run_yolo_robustness.py if corrupted data exists",
    )
    args = parser.parse_args()

    # Resolve weights list from either explicit paths or directory scan.
    try:
        weights_list = iter_weight_files(args.weights, args.weights_dir, args.pattern)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not weights_list:
        print("No weights provided/found. Use --weights /path/to/best.pt or --weights_dir <dir>.", file=sys.stderr)
        return 2
    missing = [str(p) for p in weights_list if not p.exists()]
    if missing:
        print("Missing weights files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2
    if not args.data_yaml.exists():
        print(f"Missing dataset YAML: {args.data_yaml}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for wpath in weights_list:
        print("Demo: Ultralytics val (test split)")
        print(f"  weights:  {wpath}")
        print(f"  data:     {args.data_yaml}")
        metrics = run_ultralytics_val(wpath, args.data_yaml)
        print(
            f"  P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f}"
        )

        row = {"weights": str(wpath), **metrics}
        summary_rows.append(row)

        safe_name = wpath.parent.name + "_" + wpath.name
        out_json = args.out_dir / f"val_test_metrics__{safe_name}.json"
        out_json.write_text(json.dumps(row, indent=2) + "\n")
        print(f"Wrote {out_json.relative_to(PROJECT_ROOT)}")

    if args.run_robustness:
        corruption_root = PROJECT_ROOT / "data" / "processed_corrupted"
        if not corruption_root.exists():
            print(f"Skip robustness: missing {corruption_root}")
            return 0
        for wpath in weights_list:
            print("Demo: robustness Step 2 (YOLO predict + greedy IoU Step 4)")
            print(f"  weights:  {wpath}")
            rc = run_robustness_step2(wpath)
            if rc != 0:
                print(f"Robustness Step 2 failed with exit code {rc}", file=sys.stderr)
                return rc

    out_csv = args.out_dir / "val_test_metrics_summary.csv"
    write_summary_csv(out_csv, summary_rows)
    print(f"Wrote {out_csv.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

