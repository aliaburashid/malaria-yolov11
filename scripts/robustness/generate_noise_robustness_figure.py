"""
Plot end-to-end F1 vs Gaussian noise severity from robustness CSVs.

Reads e2e_f1 from:
  runs/robustness/yolo_robustness_metrics.csv   (Step 2: YOLO predict + Step 4 greedy IoU)
  runs/robustness/two_stage_robustness_metrics.csv (Step 3: two-stage + Step 4 greedy IoU)

Rows used: clean, noise_mild, noise_medium, noise_strong (x-order).

Output:
  runs/robustness/robustness_noise_figure.png  (150 dpi)

Visual style matches the dissertation template (same as scripts/crowded_field/
generate_robustness_figure.py): axis labels, top-right legend, serif fonts, no figure title.

Run from project root:
  python3 scripts/robustness/generate_noise_robustness_figure.py

References:
- matplotlib.pyplot.plot API:
  https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
- csv.DictReader (column access by header name):
  https://docs.python.org/3/library/csv.html#csv.DictReader
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

# Matplotlib cache: avoid writing under ~/.matplotlib when it is not writable (CI/sandbox).
_repo_root = Path(__file__).resolve().parent.parent.parent
_matplotlib_cache = _repo_root / ".matplotlib-cache"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib.pyplot as plt

YOLO_METRICS_CSV = _repo_root / "runs" / "robustness" / "yolo_robustness_metrics.csv"
TWO_STAGE_METRICS_CSV = _repo_root / "runs" / "robustness" / "two_stage_robustness_metrics.csv"
OUTPUT_FIGURE = _repo_root / "runs" / "robustness" / "robustness_noise_figure.png"

# X-axis order and tick labels (must match condition names in the CSV "condition" column).
NOISE_CONDITION_ORDER = ["clean", "noise_mild", "noise_medium", "noise_strong"]
TICK_LABELS: List[str] = ["Clean", "Mild", "Medium", "Strong"]


def load_e2e_by_condition(csv_path: Path) -> Dict[str, float]:
    """Map condition string -> end-to-end F1 from the e2e_f1 column."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")

    by_condition: Dict[str, float] = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("condition", "").strip()
            raw = row.get("e2e_f1")
            if not name or raw is None or raw == "":
                continue
            by_condition[name] = float(raw)
    return by_condition


def series_for_noise_plot(by_condition: Dict[str, float]) -> List[float]:
    """Return four y-values in clean -> mild -> medium -> strong order."""
    out: List[float] = []
    for key in NOISE_CONDITION_ORDER:
        if key not in by_condition:
            raise KeyError(f"Missing condition {key!r} in CSV (have: {sorted(by_condition)})")
        out.append(by_condition[key])
    return out


def main() -> None:
    yolo_map = load_e2e_by_condition(YOLO_METRICS_CSV)
    two_stage_map = load_e2e_by_condition(TWO_STAGE_METRICS_CSV)

    yolo_f1 = series_for_noise_plot(yolo_map)
    two_stage_f1 = series_for_noise_plot(two_stage_map)

    x_positions = list(range(len(NOISE_CONDITION_ORDER)))

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    # Figure size matches crowded_field script (width 8); height 5 is readable for two lines.
    figure, axes = plt.subplots(figsize=(8, 5), dpi=150)

    axes.plot(
        x_positions,
        yolo_f1,
        color="darkblue",
        linestyle="-",
        marker="o",
        markersize=7,
        linewidth=2,
        label="YOLO Condition D",
    )
    axes.plot(
        x_positions,
        two_stage_f1,
        color="orange",
        linestyle="--",
        marker="s",
        markersize=7,
        linewidth=2,
        label="Two-stage fine-tuned",
    )

    axes.set_xticks(x_positions)
    axes.set_xticklabels(TICK_LABELS)
    axes.set_xlabel("Noise severity")
    axes.set_ylabel("End-to-end F1")
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    axes.grid(axis="y", color="lightgray", linestyle="-", linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)

    axes.legend(
        loc="upper right",
        framealpha=0.95,
        fancybox=False,
        edgecolor="0.75",
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=150, format="png", bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {OUTPUT_FIGURE.relative_to(_repo_root)}")
    print("YOLO e2e_f1:", yolo_f1)
    print("Two-stage e2e_f1:", two_stage_f1)


if __name__ == "__main__":
    main()
