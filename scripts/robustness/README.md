# Robustness to image quality variation

This folder contains scripts for the **robustness experiment**: the **same test images and the same label files** are evaluated under **clean** pixels and under **controlled corruptions** (blur, brightness, contrast, noise, JPEG). Each corruption is applied at **mild**, **medium**, and **strong** strength, using the shared definitions in `corruption_definitions.py` (the same file used for Figure 4 when you match the experiment settings).

**Why this experiment matters:** The research question asks how models behave when **image quality** changes (focus, lighting, compression, sensor noise). Here, any change in metrics is caused by **appearance**, not by a different split or relabelling—so **drops vs clean** are easy to interpret.

The explanations below use the same style as `scripts/class_imbalance/README.md` and `scripts/two_stage_baseline/README.md`: what each step does, then **results tables** you can fill from the CSVs after a full run.

---

## What we corrupt (and how strong)

Parameters are fixed in code so runs stay **reproducible** and match the batch folders created in Step 1.

| Corruption | Mild | Medium | Strong | What it simulates (informally) |
|------------|------|--------|--------|--------------------------------|
| **blur** | radius 1.5 | 3.0 | 5.0 | Defocus / motion blur |
| **brightness** | factor 0.85 | 0.65 | 0.45 | Under-exposure / darker field |
| **contrast** | factor 0.85 | 0.60 | 0.40 | Weak staining / flat appearance |
| **noise** | std 15 | 35 | 60 | Sensor or acquisition grain |
| **jpeg** | quality 75 | 50 | 25 | Lossy storage / transmission |

Folder names on disk are `clean`, then `{corruption}_{level}` (e.g. `blur_mild`, `jpeg_strong`).

---

## What we measure

### End-to-end YOLO (Step 2)

The script runs `model.val()` on each folder’s `dataset.yaml` (test split only) and records:

| Metric | Meaning |
|--------|--------|
| **P** | Precision (overall) |
| **R** | Recall (overall) |
| **F1** | Harmonic mean of P and R |
| **mAP50** | Mean AP at IoU ≥ 0.5 |
| **mAP50-95** | Mean AP averaged over IoU 0.50–0.95 |

### Two-stage pipeline (Step 3)

For each image folder, the script runs two-stage inference then evaluation and records:

| Metric | Meaning |
|--------|--------|
| **detection_f1** | F1 for box matching (IoU ≥ 0.5) |
| **e2e_f1** | End-to-end F1 (correct box **and** correct infection label) |
| **cls_accuracy** | Classification accuracy on matched crops |

### Drops (Step 4)

**Drop** = metric on corrupted − metric on **clean** (e.g. clean F1 0.91 → blur F1 0.84 → drop **−0.07**). Step 4 prints comparison tables and writes `robustness_drops_summary.csv`.

---

## What happens when the pipeline runs

The pipeline follows these steps in order.

### Step 1 — Build clean and corrupted test sets

**Script:** `step1_create_corrupted_test_sets.py`

**What it does:**

- Reads test images from `data/processed/images/test/` and labels from `data/processed/labels/test/`.
- Writes a **`clean`** copy under `data/processed_corrupted/clean/` (same pixels as the original test set, same folder layout YOLO expects).
- For each corruption and severity, writes a sibling folder under `data/processed_corrupted/` (e.g. `data/processed_corrupted/blur_medium/images/test/`).
- Copies label `.txt` files **unchanged** so boxes and classes stay identical.
- Writes a small **`dataset.yaml`** in each folder so Ultralytics can run `val` on that folder’s test images.

**Output root:** `data/processed_corrupted/`

---

### Step 2 — End-to-end YOLO on every condition

**Script:** `step2_run_yolo_robustness.py`

**What it does:**

- Loops over **every** subfolder of `data/processed_corrupted/` that contains `dataset.yaml` (including `clean`).
- Loads one YOLO weights file (default: `runs/detect/malaria_oversampled_weighted/weights/best.pt` — Condition **D** from the class-imbalance study).
- Runs validation on the **test** split for each condition and collects P, R, F1, mAP50, mAP50-95.

**Output file:** `runs/robustness/yolo_robustness_metrics.csv`

---

### Step 3 — Two-stage (YOLO + CNN) on every condition

**Script:** `step3_run_two_stage_robustness.py`

**What it does:**

- For each condition folder under `data/processed_corrupted/`, runs `scripts/two_stage_baseline/step3_two_stage_inference.py` on the **test** images with a **unique `--suffix`** (the folder name, e.g. `blur_mild`) so predictions do not overwrite each other.
- Runs `step4_evaluate_two_stage.py` for that suffix and **parses** Detection F1, End-to-end F1, and Classification accuracy from the printed summary.

**Defaults:** YOLO weights same as Step 2; classifier `runs/classifier_27k_finetuned/best.pt` (override with `--classifier` if needed).

**Output file:** `runs/robustness/two_stage_robustness_metrics.csv`

---

### Step 4 — Report drops vs clean

**Script:** `step4_report_robustness.py`

**What it does:**

- Loads `yolo_robustness_metrics.csv` and `two_stage_robustness_metrics.csv`.
- Finds the **clean** row as the baseline.
- Prints tables of metrics and **Δ vs clean** for each corrupted condition.
- Saves a long-form summary to `runs/robustness/robustness_drops_summary.csv`.

---

## Pipeline steps and scripts (summary)

| Step | Script | What it does |
|------|--------|----------------|
| **1** | `step1_create_corrupted_test_sets.py` | Builds `clean` + all corrupted test folders under `data/processed_corrupted/`; labels unchanged. |
| **2** | `step2_run_yolo_robustness.py` | Runs YOLO `val` on each folder; saves P, R, F1, mAP50, mAP50-95 to CSV. |
| **3** | `step3_run_two_stage_robustness.py` | Runs two-stage inference + evaluation per folder; saves Detection F1, E2E F1, Cls accuracy to CSV. |
| **4** | `step4_report_robustness.py` | Computes and prints **drops** vs clean; saves `robustness_drops_summary.csv`. |

---

## Results

Tables below are copied from the committed CSVs in `runs/robustness/` (same numbers as after Steps **2–3**). **P**, **R**, **F1**, **mAP50**, **mAP50-95**, Detection F1, E2E F1, and classification accuracy are **rounded to two decimals**; **Δ F1** and **Δ E2E F1** are **three decimals** vs clean so small changes on blur/JPEG are visible.

**Weights used for this run:** end-to-end YOLO = `runs/detect/malaria_oversampled_weighted/weights/best.pt` (Condition **D**); two-stage = same YOLO + `runs/classifier_27k_finetuned/best.pt`. If you re-run with other checkpoints, update the CSVs and refresh these tables.

### End-to-end YOLO — test set under each condition

| Condition | P | R | F1 | mAP50 | mAP50-95 | Δ F1 vs clean |
|-----------|---|---|----|-------|----------|---------------|
| clean | 0.91 | 0.92 | 0.91 | 0.96 | 0.79 | +0.000 |
| blur_mild | 0.91 | 0.92 | 0.91 | 0.96 | 0.79 | -0.001 |
| blur_medium | 0.90 | 0.90 | 0.90 | 0.95 | 0.79 | -0.013 |
| blur_strong | 0.86 | 0.86 | 0.86 | 0.91 | 0.74 | -0.057 |
| brightness_mild | 0.91 | 0.93 | 0.92 | 0.96 | 0.79 | +0.005 |
| brightness_medium | 0.91 | 0.93 | 0.92 | 0.96 | 0.79 | +0.007 |
| brightness_strong | 0.91 | 0.93 | 0.92 | 0.96 | 0.79 | +0.004 |
| contrast_mild | 0.90 | 0.93 | 0.92 | 0.96 | 0.79 | +0.005 |
| contrast_medium | 0.91 | 0.93 | 0.92 | 0.96 | 0.79 | +0.007 |
| contrast_strong | 0.91 | 0.92 | 0.92 | 0.95 | 0.79 | +0.004 |
| noise_mild | 0.72 | 0.78 | 0.75 | 0.76 | 0.57 | -0.167 |
| noise_medium | 0.47 | 0.56 | 0.51 | 0.47 | 0.27 | -0.400 |
| noise_strong | 0.36 | 0.21 | 0.26 | 0.20 | 0.08 | -0.654 |
| jpeg_mild | 0.91 | 0.92 | 0.91 | 0.96 | 0.79 | -0.001 |
| jpeg_medium | 0.91 | 0.92 | 0.91 | 0.96 | 0.79 | -0.002 |
| jpeg_strong | 0.90 | 0.92 | 0.91 | 0.95 | 0.78 | -0.004 |

**CSV:** `runs/robustness/yolo_robustness_metrics.csv`

---

### Two-stage pipeline (fine-tuned classifier) — test set

| Condition | Detection F1 | End-to-end F1 | Classification accuracy | Δ E2E F1 vs clean |
|-----------|--------------|---------------|---------------------------|-------------------|
| clean | 0.92 | 0.91 | 0.99 | +0.000 |
| blur_mild | 0.92 | 0.91 | 0.99 | +0.003 |
| blur_medium | 0.92 | 0.91 | 0.99 | +0.005 |
| blur_strong | 0.93 | 0.90 | 0.98 | -0.003 |
| brightness_mild | 0.93 | 0.92 | 0.99 | +0.009 |
| brightness_medium | 0.93 | 0.91 | 0.98 | +0.005 |
| brightness_strong | 0.94 | 0.88 | 0.94 | -0.031 |
| contrast_mild | 0.93 | 0.91 | 0.99 | +0.008 |
| contrast_medium | 0.93 | 0.91 | 0.97 | +0.001 |
| contrast_strong | 0.94 | 0.88 | 0.93 | -0.031 |
| noise_mild | 0.80 | 0.79 | 0.99 | -0.118 |
| noise_medium | 0.44 | 0.42 | 0.96 | -0.487 |
| noise_strong | 0.07 | 0.05 | 0.71 | -0.854 |
| jpeg_mild | 0.92 | 0.91 | 0.99 | +0.001 |
| jpeg_medium | 0.92 | 0.91 | 0.99 | +0.001 |
| jpeg_strong | 0.91 | 0.90 | 0.99 | -0.006 |

**CSV:** `runs/robustness/two_stage_robustness_metrics.csv`

---

## Observations

1. **Gaussian noise is the hardest corruption for both pipelines.** Strong noise cuts YOLO **F1** by about **0.65** and **mAP50** to **0.20**; medium noise still drops F1 by about **0.40**. Two-stage **noise_strong** collapses **detection F1** (**0.07**) and **end-to-end F1** (**0.05**), with classification accuracy falling to **0.71**—localisation and appearance both fail when the field is dominated by noise.

2. **Blur, brightness, contrast, and JPEG are comparatively mild** for end-to-end YOLO on this run: **F1** stays near the clean **0.91** except **blur_strong** (**0.86**, Δ about **−0.06**). Mild noise already hurts YOLO more than strong JPEG.

3. **Two-stage behaviour differs by corruption type.** **Brightness_strong** and **contrast_strong** reduce **end-to-end F1** to about **0.88** while reported **detection F1** stays high (**~0.94**), suggesting the **CNN** is more stressed by appearance change than the detector. **Noise_medium** and **noise_strong** hurt **detection** first (low detection F1), then end-to-end and classification.

4. **Blur and JPEG** leave two-stage **end-to-end F1** within a few thousandths of clean; the largest non-noise two-stage drops on this run are **brightness_strong**, **contrast_strong**, and the noise levels above.

---

## Prerequisites

- Test images and labels exist: `data/processed/images/test/`, `data/processed/labels/test/` (from the class-imbalance preprocessing pipeline).
- **YOLO** weights for the model you want to stress-test (default: Condition **D** `runs/detect/malaria_oversampled_weighted/weights/best.pt`).
- **Two-stage:** classifier weights available (default **fine-tuned** `runs/classifier_27k_finetuned/best.pt`). The two-stage baseline README describes how to train these if you have not already.

---

## How to run (from project root)

```bash
# 1. Create corrupted test sets (once per refresh of test data)
python3 scripts/robustness/step1_create_corrupted_test_sets.py

# 2. End-to-end YOLO on clean + every corruption folder
python3 scripts/robustness/step2_run_yolo_robustness.py
# Optional: different weights
# python3 scripts/robustness/step2_run_yolo_robustness.py --yolo_weights runs/detect/malaria/weights/best.pt

# 3. Two-stage on clean + every corruption folder
python3 scripts/robustness/step3_run_two_stage_robustness.py
# Optional: 27k-only classifier
# python3 scripts/robustness/step3_run_two_stage_robustness.py --classifier runs/classifier_27k/best.pt

# 4. Print drops and write summary CSV
python3 scripts/robustness/step4_report_robustness.py
```

**At the end you should have:**

- Corrupted image trees under `data/processed_corrupted/`
- `runs/robustness/yolo_robustness_metrics.csv`
- `runs/robustness/two_stage_robustness_metrics.csv`
- `runs/robustness/robustness_drops_summary.csv`

If you re-run the experiment, refresh the **Results** tables and **Observations** in this README so they stay aligned with `runs/robustness/yolo_robustness_metrics.csv` and `two_stage_robustness_metrics.csv`.

---

## What each script does

| Script | What it does |
|--------|----------------|
| **corruption_definitions.py** | Single source of truth for corruption types, severities, and PIL-based helpers (shared with Step 1 and dissertation figures). |
| **step1_create_corrupted_test_sets.py** | Builds `data/processed_corrupted/<condition>/` with `images/test`, `labels/test`, and `dataset.yaml`. |
| **step2_run_yolo_robustness.py** | Runs YOLO validation per condition; writes `yolo_robustness_metrics.csv`. |
| **step3_run_two_stage_robustness.py** | Calls two-stage Step 3 + Step 4 per condition; writes `two_stage_robustness_metrics.csv`. |
| **step4_report_robustness.py** | Loads both CSVs, prints drop tables, writes `robustness_drops_summary.csv`. |

---

For the full project layout and links to the class-imbalance and two-stage READMEs, see the main [README.md](../../README.md) in the project root.
