# Malaria YOLOv11 — End-to-End Detection & Classification

## Project overview & problem motivation

This project is about **improving how we detect malaria from microscope blood images**. Right now, the process is **slow and depends on experience** — experts look at slides by hand. The goal is to **automate this** by **comparing two different approaches**.

## First approach: end-to-end model

The first approach is an **end-to-end model using YOLOv11**. A **single network** does everything: it finds the red blood cells and directly classifies them as **parasitized** or **uninfected**.

## Second approach: two-stage pipeline

The second approach is a **two-stage pipeline**. First, **YOLO detects where the cells are**. Then, each detected cell is **cropped** and passed into a **separate CNN classifier** that decides if it is infected or not.

## Research question

**Is it better to do everything in one model, or to split detection and classification into two steps?**

## Evaluation & key outputs

To make the comparison fair, both approaches are **trained and evaluated on the same NIH dataset**. They are tested on:

- how **accurately** they detect cells
- how **well** they classify infection
- how **robust** they are when image quality changes (e.g. blur, noise, or lighting differences)

### Class imbalance (briefly)

One issue in the dataset is **class imbalance** — there are more healthy cells than infected ones.  
To handle this, the project tests **class weighting** and **oversampling** strategies to improve how well parasitized cells are detected.  
Full details and results are in the [class imbalance README](scripts/class_imbalance/README.md).

Results, trained models, and code are in this repo. See the sections below for setup, pipeline steps, and detailed READMEs for [class imbalance](scripts/class_imbalance/README.md) and [two-stage baseline](scripts/two_stage_baseline/README.md).

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

For step-by-step run instructions, see [class imbalance](scripts/class_imbalance/README.md) and [two-stage baseline](scripts/two_stage_baseline/README.md).

## Project structure

```
malaria-yolov11/
├── config/
│   ├── default.yaml            # Training hyperparameters, class_weights
│   ├── dataset.yaml             # YOLO dataset (Conditions A, B)
│   └── dataset_oversampled.yaml # Oversampled train list (Conditions C, D)
├── data/
│   ├── splits/                  # train/val/test patient IDs (create_splits.py)
│   ├── processed/               # YOLO images & labels (convert_to_yolo.py)
│   │   ├── images/              # train, val, test
│   │   └── labels/              # train, val, test
│   ├── cell_images/             # 27k cell dataset (two-stage classifier; Parasitized/, Uninfected/)
│   └── verify_samples/          # Sample images with boxes (verify_conversion.py)
├── scripts/
│   ├── class_imbalance/         # Data prep, training (A/B/C/D), evaluation
│   │   ├── README.md
│   │   ├── assets/              # Example verify images for README
│   │   ├── create_splits.py
│   │   ├── convert_to_yolo.py
│   │   ├── verify_conversion.py
│   │   ├── compute_class_weights.py
│   │   ├── build_oversampled_train_list.py
│   │   ├── train.py
│   │   ├── evaluate_conditions.py
│   │   └── run_publication_predictions.py
│   ├── two_stage_baseline/      # Two-stage pipeline (YOLO + CNN classifier)
│   │   ├── README.md
│   │   ├── assets/              # Pipeline and dataset diagrams
│   │   ├── step1_check_cell_images.py
│   │   ├── step2_train_classifier_27k.py
│   │   ├── step2b_finetune_classifier_thinsmear.py
│   │   ├── step3_two_stage_inference.py
│   │   └── step4_evaluate_two_stage.py
│   └── robustness/              # Image corruption experiments
│       ├── README.md
│       ├── step1_create_corrupted_test_sets.py
│       ├── step2_run_yolo_robustness.py
│       ├── step3_run_two_stage_robustness.py
│       └── step4_report_robustness.py
├── runs/
│   ├── detect/                  # YOLO training runs (malaria, malaria_weighted, ...) + CSVs + clean_predictions
│   ├── classifier_27k/          # Baseline classifier (Step 2)
│   ├── classifier_27k_finetuned/ # Fine-tuned classifier (Step 2b)
│   └── two_stage_baseline/      # Two-stage predictions (e.g. predictions_val.json)
├── results_summary.ipynb        # Tables + clean dissertation figures
├── requirements.txt
└── README.md
```

## Reproducibility

- Random seed: 42 (set in `create_splits.py` and `config/default.yaml`)
- Splits are by patient to avoid data leakage
- Config files record hyperparameters

---

## Dataset and attribution

The thin blood smear images and annotations are from the **NIH-NLM Thin Blood Smears (P. falciparum)** dataset.

**Source:** National Library of Medicine, National Institutes of Health, Bethesda, MD, USA.

**Data:** [NIH malaria datasets](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) — Polygon Set (193 patients, manual polygon annotations of red blood cells).

We request that any publication using this data attribute the source as above and cite:

> Yasmin M. Kassim, Kannappan Palaniappan, Feng Yang, Mahdieh Poostchi, Nila Palaniappan, Richard J. Maude, Sameer Antani, Stefan Jaeger. **Clustering-Based Dual Deep Learning Architecture for Detecting Red Blood Cells in Malaria Diagnostic Smears.** *IEEE Journal of Biomedical and Health Informatics*, 2020.

RBCNet code (cell detection): https://github.com/nlm-malaria/RBCNet
