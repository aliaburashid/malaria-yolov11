# Class Imbalance Scripts

This folder contains the scripts used to prepare the data, train YOLOv11n models, and compare results for the malaria detection project.

---

## What is class imbalance?

In our dataset there are two classes:

- **Parasitized** — infected red blood cells
- **Uninfected** — healthy red blood cells

The training split contains 649 parasitized and 24,004 uninfected instances, giving a ratio of approximately 37:1. This imbalance arises from the biology of malaria infection: parasitaemia in Plasmodium falciparum typically affects fewer than 1% of red blood cells even in clinical cases.

If we train the model on this data without any fix, it may learn to predict "uninfected" most of the time, because that is the majority class. We use two kinds of fixes:

1. **Class weights** — We give more importance to parasitized cells during training so the model pays more attention to them.
2. **Oversampling** — We repeat images that contain parasitized cells in the training data so the model sees them more often.

---

## The four training conditions (A, B, C, D)

We train the same YOLOv11n model four different ways to see which method works best.

| Condition | What we do | Purpose |
|-----------|------------|---------|
| **A** | Train as usual (no weights, no oversampling) | Baseline: see how the model does without any fix |
| **B** | Use **class weights** in the loss (parasitized weight = 1.9473, uninfected = 0.0527) | So the model cares more about getting parasitized cells right |
| **C** | Use **oversampling**: images with parasitized cells appear 3x in the training list (PARASITIZED_EXTRA_REPEATS = 2) | So the model sees parasitized examples more often |
| **D** | Use **both** oversampling and class weights | Combine both fixes |

After training, we evaluate all four models on the same validation and test sets. This allows a fair comparison of which imbalance method works best.

---

## Data Splitting Strategy

The dataset is split **by patient**, not by individual images, to prevent data leakage.

| Split | Patients | Images | Cells |
|-------|----------|--------|-------|
| Training | 23 | 115 | 24,653 |
| Validation | 4 | 20 | 4,118 |
| Test | 6 | 30 | 5,442 |
| **Total** | **33** | **165** | **34,213** |

A fixed random seed of 42 is used throughout to ensure full reproducibility.

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **TP** | Correct detection (predicted label matches ground truth at IoU >= 0.5) |
| **FP** | Unmatched prediction |
| **FN** | Unmatched ground-truth box |
| **Recall** | TP / (TP + FN) |
| **Precision** | TP / (TP + FP) |
| **F1** | Harmonic mean of precision and recall |
| **mAP50** | Mean Average Precision at IoU >= 0.5 (PASCAL VOC standard) |
| **mAP50-95** | Mean Average Precision averaged over IoU thresholds 0.50-0.95 (COCO standard) |

In malaria detection, **parasitized recall is the clinically critical metric**. A missed infected cell means a patient may not receive timely treatment.

---

## Results

### Validation Set

| Condition | Par. P | Par. R | Par. F1 | Uninf. R | mAP50 | mAP50-95 |
|-----------|--------|--------|---------|----------|-------|----------|
| A (baseline) | 1.00 | 0.00 | 0.00 | 0.97 | 0.50 | 0.44 |
| B (weighted) | 0.80 | 0.84 | 0.82 | 0.98 | 0.92 | 0.76 |
| C (oversampled) | 0.72 | 0.70 | 0.71 | 0.98 | 0.85 | 0.71 |
| D (oversampled + weighted) | 0.83 | 0.86 | 0.84 | 0.97 | 0.94 | 0.77 |

### Test Set

| Condition | Par. P | Par. R | Par. F1 | Uninf. R | mAP50 | mAP50-95 |
|-----------|--------|--------|---------|----------|-------|----------|
| A (baseline) | 0.19 | 0.36 | 0.25 | 0.99 | 0.58 | 0.48 |
| B (weighted) | 0.89 | 0.85 | 0.87 | 0.96 | 0.96 | 0.79 |
| C (oversampled) | 0.71 | 0.81 | 0.76 | 0.95 | 0.90 | 0.77 |
| D (oversampled + weighted) | 0.91 | 0.87 | 0.89 | 0.98 | 0.96 | 0.79 |

**Note on Condition A:** Validation parasitized F1 = 0.00 because the model never confidently predicts a parasitized cell on the validation split (precision = 1.00, recall = 0.00). The test set shows F1 = 0.25 due to different patient composition across splits with the same checkpoint. This split-level variability is expected given the small patient count per subset.

**Key Finding:** Condition D (oversampling + class weights) achieves the highest parasitized F1 (0.89) and mAP50 (0.96) on the test set and is selected as the YOLO model for all downstream experiments.

---

## What each script does

| Script | What it does |
|--------|--------------|
| **create_splits.py** | Splits the dataset by patient into train / val / test using seed 42. Saves patient lists to `data/splits/`. Prints overlap check confirming zero leakage. |
| **convert_to_yolo.py** | Converts NIH polygon annotations to YOLO format bounding boxes. Saves to `data/processed/`. Prints per-split image and cell counts on completion. |
| **verify_conversion.py** | Optional. Draws bounding box overlays on sample images (red = parasitized, green = uninfected) to verify conversion correctness. A label integrity check confirms every processed image has a corresponding YOLO label file across all three subsets, with zero missing labels. |
| **compute_class_weights.py** | Counts parasitized vs uninfected instances in training labels and computes inverse-frequency class weights normalised to sum to 2. Outputs weights to `config/default.yaml`. |
| **build_oversampled_train_list.py** | Builds training list where parasitized-positive images appear 3x (PARASITIZED_EXTRA_REPEATS = 2). Saves to `train_oversampled.txt`. |
| **train.py** | Trains YOLOv11n using Ultralytics Python API. Run once per condition with appropriate flags. Saves results to `runs/detect/`. |
| **evaluate_conditions.py** | Evaluates all four trained models on validation and test sets. Saves comparison tables to CSV. |

The images below confirm boxes are correctly aligned with cell boundaries and class colours are consistent with annotation labels (red = parasitized, green = uninfected).

![Verify sample 1](assets/verify_sample_1.jpg)
![Verify sample 2](assets/verify_sample_2.jpg)

---

## How to Run

Run the following from the project root (`malaria-yolov11/`).

### 1. Prepare the Data

```bash
python3 scripts/class_imbalance/create_splits.py
python3 scripts/class_imbalance/convert_to_yolo.py
```

Optional - verify annotation conversion:

```bash
python3 scripts/class_imbalance/verify_conversion.py
```

### 2. Handle Class Imbalance

```bash
python3 scripts/class_imbalance/compute_class_weights.py
python3 scripts/class_imbalance/build_oversampled_train_list.py
```

### 3. Train the Models

**Condition A - Baseline**
```bash
python3 scripts/class_imbalance/train.py
```

**Condition B - Class Weights**
```bash
python3 scripts/class_imbalance/train.py --weighted
```

**Condition C - Oversampling**
```bash
python3 scripts/class_imbalance/train.py --oversample
```

**Condition D - Oversampling + Class Weights**
```bash
python3 scripts/class_imbalance/train.py --oversample --weighted
```

Results saved to `runs/detect/malaria`, `runs/detect/malaria_weighted`, `runs/detect/malaria_oversampled`, `runs/detect/malaria_oversampled_weighted`.

### 4. Evaluate All Models

```bash
python3 scripts/class_imbalance/evaluate_conditions.py --both
```

Output files:
- `runs/detect/condition_comparison_val.csv`
- `runs/detect/condition_comparison_test.csv`

---

For the full project layout and environment setup, see the main [README.md](../../README.md) at the project root.
