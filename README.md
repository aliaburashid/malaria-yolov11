# Malaria Detection — End-to-End vs Two-Stage Pipeline

## What this project does

This project compares two approaches to automatically detecting malaria-infected cells in thin blood smear microscope images, using the NIH-NLM dataset.

**The problem:** Manual microscopy for malaria diagnosis is slow and depends heavily on operator expertise. Automated systems could support diagnosis, but it is unclear whether a single integrated model or a modular two-step approach performs better — especially when image quality is variable.

**The research question:** How does end-to-end detection-classification with YOLOv11 compare to a two-stage pipeline in terms of accuracy, localisation quality, and robustness to image quality variation?

---

## The two approaches

### Approach 1 — End-to-end model (YOLOv11n)

A single YOLOv11n network simultaneously finds cells and classifies each one as **parasitized** or **uninfected** in one forward pass.

### Approach 2 — Two-stage pipeline (YOLOv11n + ResNet-18)

The task is split into two steps:
1. YOLOv11n detects where the cells are (bounding boxes only — classification head not used)
2. Each detected cell is cropped and passed to a ResNet-18 classifier which predicts parasitized or uninfected

Both pipelines use the same YOLOv11n detector trained on the same data, so any performance difference is attributable to the classifier stage only.

---

## Key results

| Model | Detection F1 | End-to-end F1 | Matched-crop accuracy |
|-------|--------------|---------------|-----------------------|
| YOLO Condition D (end-to-end) | 0.92 | 0.86 | 0.930 |
| Two-stage baseline (27k only) | 0.92 | 0.89 | 0.973 |
| Two-stage fine-tuned | 0.92 | 0.91 | 0.988 |

**Main finding:** Both pipelines localise cells equally well. The two-stage fine-tuned pipeline outperforms end-to-end YOLO on clean images (E2E F1 = 0.91 versus 0.86), but both systems are vulnerable to strong Gaussian noise and respond differently to photometric shift.

---

## Experiments

| Experiment | What it tests | Results |
|------------|--------------|---------|
| Class imbalance (Conditions A–D) | Four strategies for handling the 37:1 parasitized/uninfected imbalance | Condition D (oversampling + class weights) achieves best parasitized F1 = 0.89 |
| Pipeline comparison | End-to-end YOLO vs two-stage pipeline on clean test images | Two-stage fine-tuned wins by 0.05 E2E F1 |
| Robustness | Both pipelines on 15 corrupted test conditions (5 types × 3 severities) | Both robust to blur/JPEG; both fail under strong noise; diverge under photometric shift |
| Crowded vs sparse fields | Whether cell density affects performance | Neither pipeline struggles — positive deltas explained by image composition |

---

## Dataset

| Split | Patients | Images | Cells |
|-------|----------|--------|-------|
| Training | 23 | 115 | 24,653 |
| Validation | 4 | 20 | 4,118 |
| Test | 6 | 30 | 5,442 |
| **Total** | **33** | **165** | **34,213** |

Two datasets are used:
- **NIH-NLM thin blood smear dataset** (Polygon Set, 33 patients) — detector training and full pipeline evaluation
- **NIH-NLM 27k cropped cell dataset** (27,558 images) — classifier training only

All splits are by patient using seed 42 to prevent data leakage.

---

## Setup

### 1. Install dependencies

```bash
cd malaria-yolov11
pip install -r requirements.txt
```

### 2. Dataset location

Place the NIH dataset in the parent folder:

```
Dissertation/
├── malaria-yolov11/     ← this project
└── NIH-NLM-ThinBloodSmearsPf/
    └── Polygon Set/
```

---

## Running the experiments

Each experiment has its own folder with a README and numbered scripts. Run them in order from the project root.

| Experiment | Folder | Start here |
|------------|--------|-----------|
| Class imbalance | `scripts/class_imbalance/` | [README](scripts/class_imbalance/README.md) |
| Two-stage pipeline | `scripts/two_stage_baseline/` | [README](scripts/two_stage_baseline/README.md) |
| Robustness | `scripts/robustness/` | [README](scripts/robustness/README.md) |
| Crowded field | `scripts/crowded_field/` | [README](scripts/crowded_field/README.md) |
| Evaluate checkpoints | `scripts/demo/` | [README](scripts/demo/README.md) |

---

## Project structure

```
malaria-yolov11/
├── config/
│   ├── default.yaml                  # Training hyperparameters and class weights
│   ├── dataset.yaml                  # YOLO dataset config (Conditions A, B)
│   └── dataset_oversampled.yaml      # Oversampled train list (Conditions C, D)
├── data/
│   ├── splits/                       # Patient ID lists (train/val/test)
│   ├── processed/                    # YOLO-format images and labels
│   │   ├── images/                   # train, val, test
│   │   └── labels/                   # train, val, test
│   ├── cell_images/                  # 27k cropped cell dataset (Parasitized/, Uninfected/)
│   ├── processed_corrupted/          # Corrupted test sets for robustness experiment
│   ├── crowded_field/                # Cell count CSV and crowded/sparse path lists
│   └── verify_samples/               # Annotation verification overlays
├── scripts/
│   ├── class_imbalance/              # Data prep, training (A/B/C/D), evaluation
│   ├── two_stage_baseline/           # Two-stage pipeline (detector + classifier)
│   ├── robustness/                   # Image corruption experiments
│   ├── crowded_field/                # Crowded vs sparse field evaluation
│   └── demo/                         # Evaluate saved checkpoints
├── runs/
│   ├── detect/                       # YOLO training outputs per condition
│   ├── classifier_27k/               # Baseline classifier checkpoint
│   ├── classifier_27k_finetuned/     # Fine-tuned classifier checkpoint
│   ├── two_stage_baseline/           # Two-stage predictions and evaluation outputs
│   ├── robustness/                   # Robustness experiment CSVs
│   ├── crowded_field/                # Crowded field evaluation CSVs
│   └── demo/                         # Demo metrics outputs
├── requirements.txt
└── README.md
```

---

## Reproducibility

- Random seed: 42 throughout all scripts
- Patient-level splits prevent data leakage
- All hyperparameters recorded in `config/default.yaml`
- Full environment specification in `requirements.txt`

---

## Dataset attribution

**NIH-NLM Thin Blood Smears (Plasmodium falciparum)**
National Library of Medicine, National Institutes of Health, Bethesda, MD, USA.

Data available at: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets

If you use this data, please cite:

> Kassim, Y.M., Palaniappan, K., Yang, F., Poostchi, M., Palaniappan, N., Maude, R.J., Antani, S. and Jaeger, S. 2021. Clustering-based dual deep learning architecture for detecting red blood cells in malaria diagnostic smears. *IEEE Journal of Biomedical and Health Informatics*, 25(5), pp.1735–1746.
