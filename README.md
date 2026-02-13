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

### 3. Run pipeline

From the project root (`malaria-yolov11/`):

```bash
# Step 1: Create patient-level train/val/test splits (70/15/15)
python scripts/create_splits.py

# Step 2: Convert NIH annotations to YOLO format
python scripts/convert_to_yolo.py

# Step 3: Verify conversion (optional – draws boxes on sample images)
python scripts/verify_conversion.py
# Check data/verify_samples/ – red=parasitized, green=uninfected

# Step 4: Train YOLOv11
python scripts/train.py
```

## Project structure

```
malaria-yolov11/
├── config/
│   ├── default.yaml      # Training hyperparameters
│   └── dataset.yaml      # YOLO dataset config
├── data/
│   ├── splits/           # train/val/test patient IDs (created by create_splits.py)
│   ├── processed/        # YOLO-format images & labels (created by convert_to_yolo.py)
│   └── verify_samples/   # Sample images with drawn boxes (verify_conversion.py)
├── scripts/
│   ├── create_splits.py
│   ├── convert_to_yolo.py
│   ├── verify_conversion.py
│   └── train.py
├── runs/                 # Training outputs (created by train.py)
├── requirements.txt
└── README.md
```

## Reproducibility

- Random seed: 42 (set in `create_splits.py` and `config/default.yaml`)
- Splits are by patient to avoid data leakage
- Config files record hyperparameters

## Research question

"How does end-to-end detection-classification with YOLOv11 compare to a two-stage pipeline (separate detection and classification networks) in terms of accuracy, localisation quality, and robustness to image quality variation?"
