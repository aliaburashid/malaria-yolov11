# Malaria YOLOv11 — End-to-End Detection & Classification

Dissertation project: comparing end-to-end YOLOv11 with two-stage pipelines for malaria parasite detection on NIH thin blood smears.

## Setup

### 1. Environment

```bash
cd malaria-yolov11
pip install -r requirements.txt
```

### 2. Dataset

Ensure the NIH dataset is in the parent folder:

```
Dissertation/
├── malaria-yolov11/     # this project
└── NIH-NLM-ThinBloodSmearsPf/
    └── Polygon Set/
```

### 3. Code pipeline

From the project root (`malaria-yolov11/`). Use `python3` on macOS if `python` is not available.

**Data preparation (run once)**

```bash
# 1. Patient-level train/val/test splits (70/15/15)
python3 scripts/create_splits.py

# 2. Convert NIH polygon annotations to YOLO format
python3 scripts/convert_to_yolo.py

# 3. [Optional] Verify conversion – draws boxes on sample images
python3 scripts/verify_conversion.py
# → Check data/verify_samples/; boxes are drawn with class labels (parasitized/uninfected)
```

**Class weights and oversampling (for Conditions B, C, D)**

```bash
# 4a. [For B and D] Compute class weights from train labels; update config/default.yaml if needed
python3 scripts/compute_class_weights.py

# 4b. [For C and D] Build oversampled train list (parasitized images repeated)
python3 scripts/build_oversampled_train_list.py
```

**Training (Conditions A → B → C → D)**

For **A and B**, set `config/default.yaml` → `training.name` to the condition name; the script uses that. For **C and D**, the script auto-names the run (`malaria_oversampled` / `malaria_oversampled_weighted`), so no config change is required.

```bash
# 5a. Condition A – baseline
# In config/default.yaml set: name: "malaria"
python3 scripts/train.py

# 5b. Condition B – weighted loss only (weights applied by default from config)
# In config/default.yaml set: name: "malaria_weighted"
python3 scripts/train.py

# 5c. Condition C – oversampling only (script auto-names: malaria_oversampled)
python3 scripts/train.py --oversample

# 5d. Condition D – oversampling + weighted (script auto-names: malaria_oversampled_weighted)
python3 scripts/train.py --oversample --weighted
```

**Evaluation**

```bash
# 6. Compare A/B/C/D on val and test; writes condition_comparison_val.csv and condition_comparison_test.csv
python3 scripts/evaluate_conditions.py --both
# Results in runs/detect/condition_comparison_val.csv and condition_comparison_test.csv
```

**Dissertation figures (optional)**

```bash
# 7. Clean prediction figures (test images with parasitized cells) → runs/detect/clean_predictions/
# Run the “Clean dissertation figures” cell in results_summary.ipynb
# Or: python3 scripts/run_publication_predictions.py  (if configured for clean_predictions)
```

**Pipeline summary**

| Step | Script | Output / purpose |
|------|--------|-------------------|
| 1 | `create_splits.py` | `data/splits/` (train/val/test patient IDs) |
| 2 | `convert_to_yolo.py` | `data/processed/` (images + labels) |
| 3 | `verify_conversion.py` | `data/verify_samples/` (optional sanity check) |
| 4a | `compute_class_weights.py` | Class weights for config (B, D) |
| 4b | `build_oversampled_train_list.py` | Oversampled train list (C, D) |
| 5 | `train.py` (×4 for A/B/C/D) | `runs/detect/malaria*` (weights + metrics) |
| 6 | `evaluate_conditions.py --both` | `condition_comparison_val.csv`, `condition_comparison_test.csv` |
| 7 | `results_summary.ipynb` (cell) | `runs/detect/clean_predictions/` (figures) |

## Project structure

```
malaria-yolov11/
├── config/
│   ├── default.yaml           # Training hyperparameters, class_weights
│   ├── dataset.yaml            # YOLO dataset (Conditions A, B)
│   └── dataset_oversampled.yaml # Oversampled train list (Conditions C, D)
├── data/
│   ├── splits/                 # train/val/test patient IDs (create_splits.py)
│   ├── processed/              # YOLO images & labels (convert_to_yolo.py)
│   └── verify_samples/         # Sample images with boxes (verify_conversion.py)
├── scripts/
│   ├── create_splits.py
│   ├── convert_to_yolo.py
│   ├── verify_conversion.py
│   ├── compute_class_weights.py
│   ├── build_oversampled_train_list.py
│   ├── train.py
│   ├── evaluate_conditions.py
│   ├── run_publication_predictions.py
│   └── ...
├── runs/detect/                # Training runs (malaria, malaria_weighted, ...) + CSVs + clean_predictions
├── results_summary.ipynb       # Tables + clean dissertation figures
├── requirements.txt
└── README.md
```

## Reproducibility

- Random seed: 42 (set in `create_splits.py` and `config/default.yaml`)
- Splits are by patient to avoid data leakage
- Config files record hyperparameters

## Research question

"How does end-to-end detection-classification with YOLOv11 compare to a two-stage pipeline (separate detection and classification networks) in terms of accuracy, localisation quality, and robustness to image quality variation?"
