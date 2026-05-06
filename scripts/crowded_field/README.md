# Crowded Field Evaluation (Test Split)

This folder contains scripts to assess whether cell density affects pipeline performance, comparing **YOLO Condition D** and the **two-stage fine-tuned pipeline** on **crowded** versus **sparse** test images.

Crowding is defined purely from **ground-truth cell counts** — no model predictions are used in the split.

---

## Motivation

Two reasons motivated this experiment:

1. Feedback from an NHS clinical microscopist (see dissertation Appendix E) who noted that crowded and imperfect fields are commonly encountered in routine laboratory practice.
2. The assumption embedded in the greedy matching protocol that thin blood smear cells are small and largely non-overlapping. If that assumption breaks down at higher densities, the two-stage pipeline may be disproportionately affected since detection errors in crowded fields cascade into classification failures.

---

## Definitions

- **Cell count per image:** number of non-empty lines in `data/processed/labels/test/<stem>.txt`
- **Threshold:** median cell count across all 30 test images = **185**
- **Crowded:** count ≥ 185 → 15 images
- **Sparse:** count < 185 → 15 images

Split list files (`test_crowded.txt`, `test_sparse.txt`) store one absolute image path per line so evaluation scripts resolve the correct subset.

---

## Run Order (from project root)

```bash
python3 scripts/crowded_field/step1_split_test_by_crowding.py
python3 scripts/crowded_field/step2_yolo_val_subsets.py
python3 scripts/crowded_field/step3_two_stage_subset_metrics.py
python3 scripts/crowded_field/step4_summary.py
```

`step4_summary.py` expects `subset_val_metrics.json` from step 2 and the two-stage `.txt` reports from step 3.

---

## Outputs

| Step | Output |
|------|--------|
| 1 | `data/splits/test_crowded.txt`, `data/splits/test_sparse.txt`, `data/crowded_field/cell_counts.csv` |
| 2 | `runs/detect/crowded_field_eval/yolo_crowded/`, `runs/detect/crowded_field_eval/yolo_sparse/` |
| 3 | `runs/detect/crowded_field_eval/two_stage_crowded_results.txt`, `two_stage_sparse_results.txt` |
| 4 | `runs/crowded_field/crowded_field_results.csv`, `crowded_field_summary.csv` |

---

## Evaluation Protocol

Both pipelines are evaluated using the same greedy IoU ≥ 0.5 one-to-one matching as `scripts/two_stage_baseline/step4_evaluate_two_stage.py`. No changes to the matching logic — only the image subset changes.

**YOLO:** `model.val(..., split='test', conf=0.25, imgsz=640, batch=8)`

**Two-stage:** same greedy IoU ≥ 0.5 matching imported from `step4_evaluate_two_stage.py`. End-to-end F1 requires both correct localisation and correct classification.

The primary reported metric is **end-to-end F1**, consistent with the pipeline comparison in section 4.2 of the dissertation.

---

## Results

| Pipeline | Crowded E2E F1 | Sparse E2E F1 | Δ (crowded − sparse) |
|----------|---------------|--------------|----------------------|
| YOLO Condition D | 0.871 | 0.839 | +0.032 |
| Two-stage fine-tuned | 0.914 | 0.899 | +0.014 |

**CSV:** `runs/crowded_field/crowded_field_results.csv`

---

## Key Observations

**1. Neither pipeline struggles on crowded fields.** Both perform slightly better on denser images than sparse ones. YOLO shows Δ = +0.032 and the two-stage pipeline Δ = +0.014.

**2. The positive deltas are explained by image composition, not genuine density robustness.** Crowded images contain more annotated cells per image, providing more true positive opportunities and naturally pushing F1 higher even if per-cell accuracy is unchanged.

**3. The two-stage pipeline maintains its absolute advantage over YOLO under both density conditions** (0.914 versus 0.871 crowded; 0.899 versus 0.839 sparse), consistent with the pipeline comparison results in Table 7b.

**4. The non-overlapping assumption holds in practice.** The greedy one-to-one matching protocol was designed with the assumption that thin blood smear cells are small and largely non-overlapping. The crowded field results confirm this assumption holds across the density range in this test set — field crowding is not a failure mode for either pipeline.

**5. Image quality variability is the more relevant practical challenge.** The NHS microscopist feedback (Appendix E) highlighted that variable image quality rather than cell density is more commonly encountered in routine laboratory settings — consistent with the robustness findings in section 4.3 of the dissertation.

---

## Constraints

- Median threshold of 185 is confirmed from `count_cells_per_image.py` output on the current 30-image test set
- Both subsets contain exactly 15 images each (median split)
- Seed 42 used throughout for reproducibility
- YOLO weights: `runs/detect/malaria_oversampled_weighted/weights/best.pt` (Condition D)
- Classifier weights: `runs/classifier_27k_finetuned/best.pt`

---

## Script Summary

| Script | What it does |
|--------|--------------|
| `step1_split_test_by_crowding.py` | Counts GT cells per test image, computes median (185), writes crowded/sparse path lists and `cell_counts.csv` |
| `step2_yolo_val_subsets.py` | Runs YOLO val on crowded and sparse subsets; saves metrics |
| `step3_two_stage_subset_metrics.py` | Runs two-stage inference and greedy IoU evaluation on each subset with unique suffix to prevent file overwriting |
| `step4_summary.py` | Loads both pipeline results, prints comparison table, saves `crowded_field_results.csv` |

---

For the full project layout see the main [README.md](../../README.md) at the project root.
