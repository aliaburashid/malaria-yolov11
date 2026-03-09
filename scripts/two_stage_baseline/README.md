# Two-Stage Baseline

This folder contains scripts for the **two-stage pipeline** used in the malaria detection project. This approach splits the task into **two separate steps** instead of a single end-to-end model.

## What I did in this stage

I built a **second way** to solve the malaria detection task, using **two models instead of one.** The pipeline works in **two steps.**

**Step 1 — Detection:** A YOLO model is used to find where the red blood cells are in the image and draw bounding boxes around them.

**Step 2 — Classification:** Each detected cell is cropped and then passed to a CNN classifier. The CNN classifier predicts one of two states for each cell: **parasitized** or **uninfected.**

```
Thin smear image
↓
YOLO detects cells
↓
Crop each detected cell
↓
CNN classifier predicts: parasitized or uninfected
```

![Two-stage pipeline overview](/scripts/two_stage_baseline/assets/two_stage_pipeline.png)

This is called a two-stage pipeline.

---

## Why we use a two-stage baseline

This pipeline is used as a **baseline comparison** with the end-to-end YOLO model.

In an end-to-end model, one network performs both detection and classification at the same time.

In a two-stage pipeline, the tasks are separated:

1. find the cells
2. classify each cell

By comparing both approaches we can study:

- detection accuracy (how well cells are located)
- classification accuracy (how well infection is predicted)
- localisation quality
- robustness to different image conditions

This helps answer the research question:

> Is it better to detect and classify malaria cells using one end-to-end model, or by separating detection and classification into two stages?

---

## The two datasets used

Two different datasets are used because they serve different roles.

![Thin smear vs 27k cell dataset](/scripts/two_stage_baseline/assets/two_datasets.png)

### 1. Thin blood smear images (193)

These are large microscope images containing many red blood cells.

- Each image contains dozens or hundreds of cells
- Images come from 193 patients (NIH dataset)
- Cells are annotated with bounding boxes

These images are used to:

- train the detector (YOLO)
- run the full two-stage pipeline
- evaluate detection performance

**Role:** teaches the model where cells are in a full slide.

### 2. The 27k cell dataset

This dataset contains 27,558 individual cell images.

- each image contains one cropped cell
- each image is labelled as: **parasitized** or **uninfected**

This dataset is used only to train the classifier (Stage 2).

**Role:** teaches the classifier what infected vs uninfected cells look like.

---

## Why both datasets are needed

Each dataset teaches a different skill.

| Dataset | Purpose |
|---------|---------|
| Thin smear images | Teach YOLO where cells are in a full slide |
| 27k cell dataset | Teach the classifier how to recognise infection |

**In the final pipeline:**

- **YOLO (trained on thin smears)** finds the cells
- **CNN classifier (trained on 27k cells)** classifies each cropped cell


---

## First experiment — Basic two-stage pipeline

In the first experiment, the classifier was trained only on the 27k cell dataset.

This dataset contains clean images where each image shows a single red blood cell.  
Because the cells are centred and clearly visible, this dataset is useful for learning the visual difference between parasitized and uninfected cells.

After training the classifier, the full two-stage pipeline was run:

1. YOLO detects cells in thin smear images
2. each detected cell is cropped
3. the CNN classifier predicts whether the cell is parasitized or uninfected

The results of this pipeline were evaluated on the validation and test sets.

---

## Why we added a second experiment

During the first experiment, we noticed a difference between the data used to train the classifier and the data it receives in the pipeline.

The 27k dataset contains clean, centred cell images.  
However, in the two-stage pipeline the classifier receives crops taken from full smear images.

These crops can look different because they may contain:

- background from neighbouring cells
- slightly imperfect crop edges
- differences in lighting or colour

Because of this difference, the classifier trained only on the 27k dataset may not always perform optimally in the full pipeline.

To address this, we introduced a second experiment using **fine-tuning**.

---

## Second experiment — Fine-tuned classifier

Fine-tuning means continuing the training of an already trained model using data that is closer to the final task.

In this case:

- we start from the classifier trained on the 27k cell dataset
- we extract cell crops from the thin smear training images using the **ground-truth bounding boxes**
- we continue training the classifier on these smear-style crops

The **ground-truth bounding boxes** come from the original annotations provided in the NIH malaria dataset.  
These boxes were manually labelled and indicate the exact location of each red blood cell in the thin smear images.

Importantly, these boxes are **not produced by YOLO**. They are the official dataset labels used to train and evaluate the detector.

By cropping cells using these annotated boxes, we obtain correctly centred cell images from the smear slides. These crops are closer to the images that the classifier will see during the two-stage pipeline.

Fine-tuning on these smear-style crops helps the classifier adapt to differences between the clean 27k cell images and the crops extracted from full microscope slides.

---

## Classifier versions used

After this step we have two classifier versions.

| Model | Description |
|------|------|
| classifier_27k | trained only on the 27k cell dataset |
| classifier_27k_finetuned | trained on the 27k dataset and then fine-tuned on thin smear crops |

Comparing these two versions allows us to see whether fine-tuning improves the two-stage pipeline.

---

## Pipeline steps and scripts

The full two-stage experiment runs in the following order.

| Step | What it does | Script |
|-----|-----|-----|
| Step 1 | Check that the 27k dataset is available and organised | step1_check_cell_images.py |
| Step 2 | Train the classifier on the 27k dataset | step2_train_classifier_27k.py |
| Step 2b | Fine-tune the classifier on thin smear crops | step2b_finetune_classifier_thinsmear.py |
| Step 3 | Run the two-stage pipeline (YOLO detects → crop → classifier predicts) | step3_two_stage_inference.py |
| Step 4 | Evaluate the results using detection and classification metrics | step4_evaluate_two_stage.py |

---

## What we measure

The evaluation measures three aspects of performance.

### Detection

Did YOLO correctly locate the cell?

Metrics used:
- Precision
- Recall
- F1 score

---

### Classification

For each detected cell, did the classifier predict the correct infection label?

Metric used:
- classification accuracy

---

### End-to-end performance

A prediction is counted as correct only if:

1. the predicted bounding box overlaps the real cell (IoU ≥ 0.5)
2. the classifier predicts the correct infection label

Precision, Recall and F1 are then computed using this stricter definition.

---

## Results

The table below compares the end-to-end YOLO model with the two-stage pipeline.

| Model | Split | Detection F1 | End-to-end F1 | Classification accuracy |
|------|------|------|------|------|
| End-to-end YOLO (Condition D) | val | 0.90 | 0.90 | — |
| End-to-end YOLO (Condition D) | test | 0.91 | 0.91 | — |
| Two-stage (27k classifier only) | val | 0.90 | 0.88 | 0.98 |
| Two-stage (27k classifier only) | test | 0.92 | 0.89 | 0.97 |
| Two-stage (fine-tuned classifier) | val | 0.90 | 0.90 | 0.99 |
| Two-stage (fine-tuned classifier) | test | 0.92 | 0.91 | 0.99 |

Sources:

- YOLO results from `condition_comparison_val.csv` and `condition_comparison_test.csv`
- Two-stage results from `step4_evaluate_two_stage.py`

## Observations

Several patterns appear in the results.

### 1. Detection performance is stable

The YOLO detector performs consistently across all experiments.

- Detection F1 stays around **0.90–0.92** on both validation and test sets.
- This means the model is able to **locate most red blood cells correctly**.

---

### 2. Two-stage pipeline (without fine-tuning) performs slightly worse

When using the classifier trained **only on the 27k dataset**:

- Classification accuracy is very high (**97–98%**)
- But the **end-to-end F1 drops slightly** to **0.88–0.89**

This suggests that the classifier struggles slightly when working with **cell crops taken from full smear images**, which look different from the clean 27k cell images.

---

### 3. Fine-tuning improves the pipeline

After **fine-tuning the classifier on thin smear crops**:

- End-to-end F1 increases to **0.90–0.91**
- Classification accuracy improves to **99%**

This shows that the classifier adapts better to the **real cell crops produced by the detection stage**.

---

### Overall takeaway

- **Fine-tuning improves the two-stage pipeline**
- After fine-tuning, the **two-stage pipeline matches the performance of end-to-end YOLO**
- However, the **end-to-end model is simpler**, since detection and classification are learned in a single network.

---

## Summary

In this stage, a two-stage pipeline was implemented where YOLO first detects cells and a CNN classifier predicts whether each cell is infected.

The classifier was first trained on the 27k cell dataset, and a second experiment introduced fine-tuning using thin smear cell crops to better match the real pipeline inputs.

The results allow a direct comparison between:

- end-to-end YOLO
- the basic two-stage pipeline
- the fine-tuned two-stage pipeline

---

## How to run (from project root)

```bash
# Step 1 — Check 27k dataset
python3 scripts/two_stage_baseline/step1_check_cell_images.py

# Step 2 — Train classifier on 27k
python3 scripts/two_stage_baseline/step2_train_classifier_27k.py

# Step 2b (optional) — Fine-tune on thin smear crops
python3 scripts/two_stage_baseline/step2b_finetune_classifier_thinsmear.py

# Step 3 — Two-stage inference (baseline classifier)
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split test

# Step 3 with fine-tuned classifier (use suffix so baseline results are not overwritten)
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split val --classifier_weights runs/classifier_27k_finetuned/best.pt --suffix finetuned
python3 scripts/two_stage_baseline/step3_two_stage_inference.py --split test --classifier_weights runs/classifier_27k_finetuned/best.pt --suffix finetuned

# Step 4 — Evaluate (baseline)
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split val
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split test

# Step 4 — Evaluate (fine-tuned)
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split val --suffix finetuned
python3 scripts/two_stage_baseline/step4_evaluate_two_stage.py --split test --suffix finetuned
```

---

## Keeping baseline and fine-tuned results separate

Step 3 writes files like `predictions_val.json` and `predictions_test.json`. If you run Step 3 again with the **fine-tuned** classifier, use one of:

- **`--suffix finetuned`** — Saves `predictions_val_finetuned.json` and `predictions_test_finetuned.json` in the same folder (recommended).
- **`--output_dir runs/two_stage_baseline_finetuned`** — Saves all fine-tuned outputs in a different folder.

Then run Step 4 with `--suffix finetuned` or `--predictions <path>` so you evaluate the right file.

---
