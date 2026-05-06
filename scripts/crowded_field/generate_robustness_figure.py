"""
Generate a robustness line figure: end-to-end F1 vs noise severity.

Data source:
- Hardcoded values in this script (no external CSV input).

Output:
- runs/robustness/robustness_noise_figure.png

Reference:
- Matplotlib plotting API: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

Run from project root:
  python3 scripts/crowded_field/generate_robustness_figure.py
"""

from __future__ import annotations

# Set MPLCONFIGDIR to a writable repo-local path when home cache is unavailable.
import os
from pathlib import Path

# Writable cache when ~/.matplotlib is not available (e.g. some sandboxes).
_PROJECT = Path(__file__).resolve().parent.parent.parent
_MPL_CACHE = _PROJECT / ".matplotlib-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt

# Output figure location.
PROJECT_ROOT = _PROJECT
OUT_DIR = PROJECT_ROOT / "runs" / "robustness"
OUT_PATH = OUT_DIR / "robustness_noise_figure.png"

# X-axis labels and fixed values used in dissertation figure draft.
SEVERITY_LABELS = ["Clean", "Mild", "Medium", "Strong"]
X = list(range(len(SEVERITY_LABELS)))

YOLO_F1 = [0.91, 0.75, 0.51, 0.26]
TWOSTAGE_F1 = [0.91, 0.79, 0.42, 0.05]


def main() -> None:
    # Ensure output directory exists before rendering.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Typography/style choices for consistency with report figures.
    # Reference: https://matplotlib.org/stable/users/explain/customizing.html
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

    # Create high-resolution figure canvas.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Series 1: YOLO Condition D end-to-end F1 by severity.
    ax.plot(
        X,
        YOLO_F1,
        color="darkblue",
        linestyle="-",
        marker="o",
        markersize=7,
        linewidth=2,
        label="YOLO Condition D",
    )
    # Series 2: two-stage fine-tuned end-to-end F1 by severity.
    ax.plot(
        X,
        TWOSTAGE_F1,
        color="orange",
        linestyle="--",
        marker="s",
        markersize=7,
        linewidth=2,
        label="Two-stage fine-tuned",
    )

    ax.set_xticks(X)
    ax.set_xticklabels(SEVERITY_LABELS)
    ax.set_xlabel("Noise severity")
    ax.set_ylabel("End-to-end F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Light horizontal grid to aid reading exact y-values.
    ax.grid(axis="y", color="lightgray", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", framealpha=0.95)

    # Save PNG output and close figure to free memory in batch runs.
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
