# Two-Stage Baseline

This folder contains scripts for the **two-stage pipeline** used in the malaria detection project. This approach splits the task into **two separate steps** instead of a single end-to-end model.

---

## What this stage does

The pipeline works in two steps.

**Step 1 — Detection:** A YOLOv11n model finds where the red blood cells are in the image and draws bounding boxes around them.

**Step 2 — Classification:** Each detected cell is cropped and passed to a ResNet-18 CNN classifier. The classifier predicts one of two labels: **parasitized** or **uninfected**.

```
Thin smear image
↓
YOLOv11n detects cells (bounding boxes only — class head not used)
↓
Crop each detected cell (padding factor 0.1 on all sides)
↓
ResNet-18 classifier predicts: parasitized or uninfected
```

![Two-stage pipeline overview](assets/two_stage_pipeline.png)

The YOLO class head output is intentionally discarded. Infection labels come exclusively from the CNN classifier, making the modular structure explicit.

---

## Why a two-stage baseline

By comparing the two-stage pipeline with end-to-end YOLO under the same data split and evaluation protocol, any performance difference can be attributed to architectural design rather than experimental variation. Both pipelines use the same YOLOv11n detector trained on the same thin-smear annotations, so the pipeline comparison is attributable to the classifier stage only.

---

## The two datasets used

![Thin smear vs 27k cell dataset](assets/two_datasets.png)

| Dataset | Patients | Images | Cells | Role |
|---------|----------|--------|-------|------|
| NIH-NLM thin smear (Polygon Set) | 33 | 165 | 34,213 | Detector training and full pipeline evaluation |
| NIH-NLM 27k cropped cells | — | 27,558 | — | Classifier training only |

The 27k dataset was not used to train the thin-smear detector. This separation keeps detector training and crop-level classification as methodologically distinct components.

---

## Patient-level split

| Split | Patients | Images | Cells |
|-------|----------|--------|-------|
| Training | 23 | 115 | 24,653 |
| Validation | 4 | 20 | 4,118 |
| Test | 6 | 30 | 5,442 |

Split uses seed 42 for full reproducibility. No patient appears in more than one subset.

---

## Two classifier variants

| Model | Description |
|-------|-------------|
| `classifier_27k` | Trained only on the 27k cropped-cell dataset |
| `classifier_27k_finetuned` | Trained on 27k then fine-tuned on ground-truth crops from thin-smear training images |

Fine-tuning closes the domain gap between the clean 27k training data and the full-smear crops the classifier receives at inference time (which include surrounding background, partial neighbouring cells, and variable lighting).

---

## Evaluation protocol

Evaluation uses greedy IoU matching at threshold 0.5, applied consistently to both pipelines using `step4_evaluate_two_stage.py`. Three levels are reported:

| Level | What it measures |
|-------|-----------------|
| **Detection F1** | F1 for box matching at IoU ≥ 0.5 (localisation quality) |
| **Classification accuracy** | Proportion of matched pairs with correct CNN label (classifier quality in isolation) |
| **End-to-end F1** | F1 requiring both correct localisation and correct classification (full pipeline quality) |

Classification accuracy is computed only on matched pairs — missed detections never enter its denominator. End-to-end F1 counts missed detections as false negatives, making it a stricter and more realistic measure of system performance.

**Key counts (two-stage fine-tuned, test set):**
- Ground-truth boxes: 5,442
- Detection TP (matched pairs): 5,350
- End-to-end TP (matched + correct label): 5,288
- Gap of 62 = cells correctly localised but misclassified
- FP = 917 = 855 unmatched predictions + 62 class-mislabelled
- FN = 154 missed ground-truth boxes

---

## Results

### Validation set

| Model | Detection F1 | E2E F1 | Classification accuracy |
|-------|--------------|--------|------------------------|
| End-to-end YOLO (Condition D) | 0.90 | 0.89 | — |
| Two-stage baseline (27k only) | 0.90 | 0.88 | 0.979 |
| Two-stage fine-tuned | 0.90 | 0.90 | 0.997 |

### Test set

| Model | Detection F1 | E2E F1 | Classification accuracy |
|-------|--------------|--------|------------------------|
| End-to-end YOLO (Condition D) | 0.92 | 0.86 | 0.930 |
| Two-stage baseline (27k only) | 0.92 | 0.89 | 0.973 |
| Two-stage fine-tuned | 0.92 | 0.91 | 0.988 |

---

## Key Observations

**1. Detection F1 is consistent at 0.90–0.92 across all variants.** Both pipelines use the same YOLOv11n detector, so equivalent localisation quality is expected. Any performance difference between pipelines is therefore attributable to the classifier stage only.

**2. The gap between classification accuracy and end-to-end F1.** The baseline classifier achieves 97.3% matched-crop accuracy on the test set but only 0.89 end-to-end F1. This is because classification accuracy excludes missed detections from its denominator. A system reporting 97.3% matched-crop accuracy could still miss a clinically significant number of infected cells when evaluated end-to-end on full smear images.

**3. Fine-tuning improves both matched-crop accuracy and end-to-end F1.** Fine-tuning raises matched-crop accuracy from 97.3% to 98.8% and end-to-end F1 from 0.89 to 0.91. The gap between oracle accuracy (98.9% on ground-truth crops) and matched-crop accuracy (98.8%) confirms the fine-tuned classifier is already near its ceiling on well-formed crops. The remaining end-to-end error is almost entirely due to missed detections.

**4. The two-stage fine-tuned pipeline outperforms end-to-end YOLO** on the test set (E2E F1 = 0.91 versus 0.86, a gap of 0.05). The added complexity of a two-stage pipeline is justified when classification accuracy on well-formed crops is the priority.

**5. Per-class breakdown (fine-tuned, matched crops, test set):**
- Parasitized matched-crop recall: 92.6% (398/430)
- Uninfected matched-crop recall: 97.6% (4,890/5,012)

---

## Oracle experiment

Both classifiers were evaluated on perfect ground-truth crops, bypassing the detector entirely, to isolate the classifier ceiling from detection errors.

| Classifier | Overall accuracy | Parasitized accuracy | Uninfected accuracy |
|------------|-----------------|---------------------|---------------------|
| Baseline (27k only) | 96.7% | 99.1% | 96.5% |
| Fine-tuned | 98.9% | 95.6% | 99.2% |

Fine-tuning raises overall accuracy but reduces parasitized accuracy by 3.5 pp. The fine-tuned model trains on smear-derived crops with variable background and lighting, which improves uninfected performance but introduces confusion on the rarer parasitized class. In a screening context this trade-off warrants attention even when overall accuracy improves.

---

## Pipeline steps and scripts

| Step | Script | What it does |
|------|--------|--------------|
| Step 1 | `step1_check_cell_images.py` | Verifies the 27k dataset is available and correctly organised |
| Step 2 | `step2_train_classifier_27k.py` | Trains ResNet-18 on 27k cells. Best checkpoint saved on strictly improved val accuracy to `runs/classifier_27k/best.pt` |
| Step 2b | `step2b_finetune_classifier_thinsmear.py` | Continues training on thin-smear GT crops. Best checkpoint to `runs/classifier_27k_finetuned/best.pt` |
| Step 3 | `step3_two_stage_inference.py` | Runs YOLO detection (conf = 0.25, padding = 0.1) then CNN classification. Saves predictions as JSON |
| Step 4 | `step4_evaluate_two_stage.py` | Greedy IoU matching at 0.5; reports detection, classification, and end-to-end metrics |

---

## How to Run

```bash
# Step 1 — Check 27k dataset
python3 scripts/two_stage_baseline/step1_check_cell_images.py

# Step 2 — Train classifier on 27k
python3 scripts/two_stage_baseline/step2_train_classifier_27k.py

# Step 2b — Fine-tune on thin smear crops
python3 scripts/two_stage_baseline/step2b_finetune_classifier_thinsmear.py

# Step 3 — Two-stage inference (baseline classifier)
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split test

# Step 3 — Two-stage inference (fine-tuned classifier)
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val   --classifier_weights runs/classifier_27k_finetuned/best.pt --suffix finetuned
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split test   --classifier_weights runs/classifier_27k_finetuned/best.pt --suffix finetuned

# Step 4 — Evaluate baseline
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split val
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split test

# Step 4 — Evaluate fine-tuned
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split val --suffix finetuned
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split test --suffix finetuned
```

---

## Keeping baseline and fine-tuned results separate

Step 3 writes `predictions_val.json` and `predictions_test.json` by default. Use `--suffix finetuned` to save `predictions_val_finetuned.json` and `predictions_test_finetuned.json` separately. Then pass `--suffix finetuned` to Step 4 to evaluate the correct file.

Traceability was confirmed from `predictions_val.json` and `predictions_val_finetuned.json`, verifying baseline and fine-tuned runs used the correct checkpoints.

---

For the full project layout see the main [README.md](../../README.md) at the project root.
