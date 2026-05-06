# Crowded-field evaluation (test split)

Compare **YOLO Condition D** and **two-stage (fine-tuned)** behaviour on **crowded** vs **sparse** test images, where crowding is defined only from **ground-truth cell counts** (lines in each YOLO label file).

## Definitions

- **Cell count per image:** number of non-empty lines in `data/processed/labels/test/<stem>.txt`.
- **Threshold:** median cell count over all test images (expected **185** for the current 30-image test set).
- **Crowded:** count ≥ median.
- **Sparse:** count < median.

Split list files (`test_crowded.txt`, `test_sparse.txt`) store **one absolute image path per line** so Ultralytics `val()` resolves the custom test list correctly.

## Run order (from project root)

```bash
python3 scripts/crowded_field/step1_split_test_by_crowding.py
python3 scripts/crowded_field/step2_yolo_val_subsets.py
python3 scripts/crowded_field/step3_two_stage_subset_metrics.py
python3 scripts/crowded_field/step4_summary.py
```

`step4_summary.py` expects `subset_val_metrics.json` from step 2 and the two-stage `.txt` reports from step 3.

## Outputs

| Step | Output |
|------|--------|
| 1 | `data/splits/test_crowded.txt`, `data/splits/test_sparse.txt` |
| 2 | `runs/detect/crowded_field_eval/yolo_crowded/`, `runs/detect/crowded_field_eval/yolo_sparse/` (Ultralytics `val` artefacts) |
| 3 | `runs/detect/crowded_field_eval/two_stage_crowded_results.txt`, `two_stage_sparse_results.txt` |
| 4 | `runs/detect/crowded_field_eval/crowded_field_summary.csv` |

## Metrics and interpretation (dissertation)

**YOLO (summary row):** macro mean of **per-class box F1** from Ultralytics (`subset_val_metrics.json`).

**Two-stage — comparable macro row:** `per_class_macro_f1` = mean of parasitized F1 and uninfected F1, where class **c** TP means greedy IoU ≥ 0.5 match **and** `pred.cls == gt.cls == c` (same match set as global end-to-end; per-class P/R/F1 from aggregated TP/FP/FN). This is the row to compare directly to YOLO macro F1 on crowded vs sparse.

**Two-stage — other rows:** global **detection** F1 (IoU match only) and global **end-to-end** F1 (same as `step4_evaluate_two_stage.py`) remain for pipeline-internal reporting; they are not averaged the same way as YOLO’s per-class box F1.

**Crowding takeaway:** compare crowded vs sparse **within** each metric row; small deltas on the macro rows suggest neither pipeline collapses on crowded fields in this split.

## Constraints (matched to the rest of the repo)

- YOLO: `model.val(..., split='test', conf=0.25, imgsz=640, batch=8)`.
- Two-stage: same greedy **IoU ≥ 0.5** one-to-one matching as `scripts/two_stage_baseline/step4_evaluate_two_stage.py` (logic imported from that file; no edits to it).
