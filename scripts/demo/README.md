# Running the Malaria Detection Pipeline

This guide explains how to evaluate the trained models from this project. It covers three experiments: standard detection, robustness testing, and the two-stage pipeline. No machine learning background is needed to follow these steps.

---

## Before You Start

Make sure the following are in place:

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Check the dataset is present**

The processed images and labels should be under `data/processed/`. If they are not, follow the setup instructions in the main [README.md](../../README.md).

**3. Check the trained model files exist**

The trained checkpoints are saved as `best.pt` files inside the `runs/` folder. For example:
- `runs/detect/malaria_oversampled_weighted/weights/best.pt` — YOLO Condition D (end-to-end detector)
- `runs/classifier_27k_finetuned/best.pt` — fine-tuned ResNet-18 classifier

---

## What You Can Run

| Experiment | What it does |
|------------|--------------|
| **Detection validation** | Evaluates how well the detector locates and classifies cells on the test set |
| **Robustness testing** | Tests both pipelines on degraded images (blur, noise, brightness changes, etc.) |
| **Two-stage pipeline** | Runs the full detector + classifier pipeline and reports end-to-end results |

---

## Step 1 — Check What Models Are Available (Recommended First)

This prints a list of all detector and classifier checkpoints found in the `runs/` folder without running any evaluation.

```bash
python3 scripts/demo/run_demo.py --weights_dir runs/detect --pattern best.pt --list_only
```

---

## Step 2 — Run Detection Validation

Evaluates a single trained detector on the test set and prints key metrics (precision, recall, F1, mAP).

```bash
python3 scripts/demo/run_demo.py --weights runs/detect/malaria_oversampled_weighted/weights/best.pt
```

To evaluate multiple detectors at once:

```bash
python3 scripts/demo/run_demo.py --weights_dir runs/detect --pattern best.pt
```

**Output files saved to `runs/demo/`:**
- `val_test_metrics_summary.csv` — results for all evaluated checkpoints in one table
- `val_test_metrics__*.json` — individual result file per checkpoint

---

## Step 3 — Run Robustness Testing

Tests the detector on corrupted versions of the test images to see how performance changes under blur, noise, brightness reduction, contrast reduction, and JPEG compression.

**First, create the corrupted test sets** (only needs to be done once):
```bash
python3 scripts/robustness/step1_create_corrupted_test_sets.py
```

**Then run the robustness evaluation:**
```bash
python3 scripts/demo/run_demo.py   --weights runs/detect/malaria_oversampled_weighted/weights/best.pt   --run_robustness
```

Results are saved to `runs/robustness/yolo_robustness_metrics.csv`. See `scripts/robustness/README.md` for the full results table and interpretation.

---

## Step 4 — Run the Two-Stage Pipeline

Runs the full two-stage pipeline: the detector finds cells, then the classifier labels each one as parasitized or uninfected. Reports detection F1, end-to-end F1, and matched-crop classification accuracy.

```bash
python3 scripts/demo/run_demo.py   --weights_dir runs/detect --pattern best.pt   --run_two_stage   --classifier_weights runs/classifier_27k_finetuned/best.pt   --two_stage_split test
```

**What the output means:**

| Metric | What it measures |
|--------|-----------------|
| Detection F1 | How well the detector finds cells (localisation only) |
| Classification accuracy | How accurately the classifier labels detected cells |
| End-to-end F1 | Combined score — a prediction is only correct if the cell is both found and labelled correctly |

Results are saved as prediction JSON files in `runs/two_stage_baseline/`.

---

## Summary of All Output Files

| File | What it contains |
|------|-----------------|
| `runs/demo/val_test_metrics_summary.csv` | Detection metrics for all evaluated checkpoints |
| `runs/demo/val_test_metrics__*.json` | Per-checkpoint detection results |
| `runs/robustness/yolo_robustness_metrics.csv` | YOLO performance under each corruption condition |
| `runs/two_stage_baseline/predictions_*_demo_*.json` | Two-stage pipeline predictions per checkpoint |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Missing weights files` | Check the path to `best.pt` — use the actual file path, not a placeholder |
| No checkpoints found | Check that `--weights_dir` points to a folder containing `best.pt` files |
| Robustness step skipped | Run `step1_create_corrupted_test_sets.py` first to generate the corrupted test folders |
| Two-stage classifier missing | Pass a valid path with `--classifier_weights` |

---

For the full project layout and experiment details see the main [README.md](../../README.md) at the project root.
