# Robustness to Image Quality Variation

This folder contains scripts for the **robustness experiment**: the **same test images and the same label files** are evaluated under **clean** pixels and under **controlled corruptions** (blur, brightness, contrast, noise, JPEG). Each corruption is applied at **mild**, **medium**, and **strong** strength, using the shared definitions in `corruption_definitions.py`.

**Why this experiment matters:** The research question asks how models behave when **image quality** changes. Here, any change in metrics is caused by **appearance only**, not by a different split or relabelling — so **drops vs clean** are easy to interpret.

---

## What we corrupt (and how strong)

Parameters are fixed in `corruption_definitions.py` so runs stay **reproducible**.

| Corruption | Mild | Medium | Strong | What it simulates |
|------------|------|--------|--------|-------------------|
| **blur** | radius 1.5 | 3.0 | 5.0 | Defocus / motion blur |
| **brightness** | factor 0.85 | 0.65 | 0.45 | Under-exposure / darker field |
| **contrast** | factor 0.85 | 0.60 | 0.40 | Weak staining / flat appearance |
| **noise** | std 15 | 35 | 60 | Sensor or acquisition grain |
| **jpeg** | quality 75 | 50 | 25 | Lossy storage / transmission |

Folder names on disk: `clean`, then `{corruption}_{level}` (e.g. `blur_mild`, `jpeg_strong`). 16 folders total (1 clean + 5 corruptions × 3 severities).

---

## What we measure

### End-to-end YOLO (Step 2)

Uses `YOLO.predict()` on each folder followed by greedy IoU matching via `step4_evaluate_two_stage.py` to ensure metrics are consistent with the two-stage evaluation protocol. **Note:** `model.val()` is not used here because it produces mAP-style metrics not aligned with the greedy end-to-end F1 protocol.

| Metric | Meaning |
|--------|---------|
| **Detection F1** | F1 for box matching at IoU >= 0.5 |
| **E2E F1** | End-to-end F1 (correct box and correct infection label) |

### Two-stage pipeline (Step 3)

| Metric | Meaning |
|--------|---------|
| **detection_f1** | F1 for box matching at IoU >= 0.5 |
| **e2e_f1** | End-to-end F1 (correct box and correct infection label) |
| **cls_accuracy** | Classification accuracy on matched crops |

### Drops (Step 4)

**Δ** = corrupted metric − clean metric. Negative = degradation, positive = better than clean.

---

## Results

**Weights used:** YOLO = `runs/detect/malaria_oversampled_weighted/weights/best.pt` (Condition D); two-stage = same YOLO + `runs/classifier_27k_finetuned/best.pt`.

**Clean baselines:** YOLO E2E F1 = 0.857, Two-stage E2E F1 = 0.907.

### End-to-end YOLO — test set

| Condition | E2E F1 | Δ E2E F1 vs clean |
|-----------|--------|-------------------|
| clean | 0.857 | +0.000 |
| blur_mild | 0.859 | +0.002 |
| blur_medium | 0.857 | +0.000 |
| blur_strong | 0.853 | −0.004 |
| brightness_mild | 0.869 | +0.012 |
| brightness_medium | 0.881 | +0.024 |
| brightness_strong | 0.888 | +0.031 |
| contrast_mild | 0.868 | +0.011 |
| contrast_medium | 0.882 | +0.025 |
| contrast_strong | 0.892 | +0.035 |
| noise_mild | 0.628 | −0.229 |
| noise_medium | 0.357 | −0.500 |
| noise_strong | 0.048 | −0.809 |
| jpeg_mild | 0.857 | +0.000 |
| jpeg_medium | 0.859 | +0.002 |
| jpeg_strong | 0.851 | −0.006 |

**CSV:** `runs/robustness/yolo_robustness_metrics.csv`

---

### Two-stage pipeline (fine-tuned classifier) — test set

| Condition | Detection F1 | E2E F1 | Cls accuracy | Δ E2E F1 vs clean |
|-----------|--------------|--------|--------------|-------------------|
| clean | 0.920 | 0.907 | 0.988 | +0.000 |
| blur_mild | 0.920 | 0.911 | 0.988 | +0.004 |
| blur_medium | 0.920 | 0.912 | 0.988 | +0.005 |
| blur_strong | 0.920 | 0.904 | 0.987 | −0.003 |
| brightness_mild | 0.930 | 0.916 | 0.986 | +0.009 |
| brightness_medium | 0.930 | 0.912 | 0.980 | +0.005 |
| brightness_strong | 0.936 | 0.876 | 0.935 | −0.031 |
| contrast_mild | 0.930 | 0.915 | 0.984 | +0.008 |
| contrast_medium | 0.930 | 0.908 | 0.970 | +0.001 |
| contrast_strong | 0.938 | 0.876 | 0.930 | −0.031 |
| noise_mild | 0.800 | 0.789 | 0.988 | −0.118 |
| noise_medium | 0.436 | 0.420 | 0.965 | −0.487 |
| noise_strong | 0.075 | 0.053 | 0.710 | −0.854 |
| jpeg_mild | 0.920 | 0.908 | 0.988 | +0.001 |
| jpeg_medium | 0.920 | 0.908 | 0.988 | +0.001 |
| jpeg_strong | 0.914 | 0.901 | 0.988 | −0.006 |

**CSV:** `runs/robustness/two_stage_robustness_metrics.csv`

---

## Key Observations

1. **Gaussian noise is the dominant failure mode for both pipelines.** At strong noise, YOLO E2E F1 collapses to 0.048 (Δ = −0.809) and two-stage to 0.053 (Δ = −0.854). Detection F1 at strong noise is 0.075 for both pipelines, confirming the shared YOLOv11n backbone can no longer localise cells. Since no classifier can recover cells the detector fails to find, the two-stage pipeline's dedicated ResNet-18 offers no protection at this severity.

2. **At mild noise, the two-stage pipeline retains more performance than YOLO** (0.789 versus 0.628, gap = 0.161). Detection F1 at mild noise is 0.800 for both, meaning the detector is still localising most cells. The advantage comes from classifying individual crops in isolation rather than simultaneously localising and classifying from a noisy image. As detection F1 collapses at medium (0.436) and strong (0.075) noise, the gap closes to 0.063 and then 0.005.

3. **Blur and JPEG produce negligible degradation.** At strong blur, YOLO drops by only Δ = −0.004 and two-stage by Δ = −0.003. Strong JPEG reduces both by Δ = −0.006. Small positive deltas at mild and medium blur reflect a threshold-shift artefact at fixed confidence 0.25 rather than genuine improvement.

4. **Brightness and contrast show opposite responses between pipelines at strong severity.** YOLO E2E F1 rises by Δ = +0.031 (brightness) and +0.035 (contrast), while the two-stage pipeline falls by Δ = −0.031 in both cases. Detection F1 is virtually identical for both at these severities (brightness strong: both 0.936; contrast strong: both 0.938), confirming the divergence comes entirely from the classification stage. When images become very dark or low-contrast, the ResNet-18 classifier defaults toward the uninfected majority class under distribution shift — a class-prior effect. YOLO's integrated head adapts more naturally.

5. **Neither pipeline struggles on clean images.** The two-stage fine-tuned pipeline achieves E2E F1 = 0.907 versus YOLO's 0.857, a gap of 0.050, consistent with the pipeline comparison results in the main dissertation.

---

## Prerequisites

- Test images and labels in `data/processed/images/test/` and `data/processed/labels/test/` (from class-imbalance preprocessing).
- YOLO Condition D weights: `runs/detect/malaria_oversampled_weighted/weights/best.pt`
- Fine-tuned classifier weights: `runs/classifier_27k_finetuned/best.pt`

---

## How to Run

```bash
# 1. Create clean and corrupted test sets (run once)
python3 scripts/robustness/step1_create_corrupted_test_sets.py

# 2. YOLO robustness evaluation
python3 scripts/robustness/step2_run_yolo_robustness.py

# 3. Two-stage robustness evaluation
python3 scripts/robustness/step3_run_two_stage_robustness.py

# 4. Compute and report drops vs clean
python3 scripts/robustness/step4_report_robustness.py
```

Output files:
- `runs/robustness/yolo_robustness_metrics.csv`
- `runs/robustness/two_stage_robustness_metrics.csv`
- `runs/robustness/robustness_drops_summary.csv`

---

## Script Summary

| Script | What it does |
|--------|--------------|
| **corruption_definitions.py** | Single source of truth for corruption types, severities, and PIL helpers. |
| **step1_create_corrupted_test_sets.py** | Builds `data/processed_corrupted/<condition>/` with images, labels, and dataset.yaml. Labels copied unchanged. |
| **step2_run_yolo_robustness.py** | Runs YOLO.predict() per condition then greedy IoU scoring; saves to `yolo_robustness_metrics.csv`. |
| **step3_run_two_stage_robustness.py** | Runs two-stage inference and evaluation per condition with unique suffix; saves to `two_stage_robustness_metrics.csv`. |
| **step4_report_robustness.py** | Loads both CSVs, computes Δ vs clean, prints tables, saves `robustness_drops_summary.csv`. |

---

For the full project layout see the main [README.md](../../README.md) at the project root.
