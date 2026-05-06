## Supervisor demo (run with `best.pt`)

This folder provides a professional, copy-paste runbook for evaluating one or more saved YOLO
checkpoints (`best.pt`) with the same repository settings used in the project.

### Prerequisites

- Python dependencies installed:

```bash
pip install -r requirements.txt
```

- Dataset present under `data/processed/` (images + labels). See repo `README.md` for dataset setup.

### Recommended run order

1. **Confirm model inventory** (`--list_only`).
2. **Run detector validation** on test split.
3. **(Optional) Run robustness** if corrupted test folders are present.
4. **(Optional) Run two-stage pipeline** with a chosen classifier checkpoint.

### 1) Inventory check (recommended first)

```bash
python3 scripts/demo/run_demo.py --weights_dir runs/detect --pattern best.pt --list_only
```

This prints:
- detector checkpoints selected for evaluation
- classifier checkpoints discovered under `runs/classifier_*/best.pt`

### 2) Quick validation run (single model)

```bash
python3 scripts/demo/run_demo.py --weights /path/to/best.pt
```

Outputs:
- Prints key metrics to stdout.
- Writes per-checkpoint JSON summaries and a combined CSV under `runs/demo/`.

### 3) Validation run (multiple models)

Provide explicit paths:

```bash
python3 scripts/demo/run_demo.py --weights /path/to/a/best.pt /path/to/b/best.pt
```

Or scan recursively:

```bash
python3 scripts/demo/run_demo.py --weights_dir runs/detect --pattern best.pt
```

### 4) Robustness run (optional)

If `data/processed_corrupted/` exists (created by `scripts/robustness/step1_create_corrupted_test_sets.py`),
the demo can run the YOLO robustness Step 2 script with the provided weights:

```bash
python3 scripts/demo/run_demo.py --weights /path/to/best.pt --run_robustness
```

References:
- `scripts/robustness/step1_create_corrupted_test_sets.py`
- `scripts/robustness/step2_run_yolo_robustness.py`

### 5) Two-stage run (optional)

Run detector + crop classifier pipeline for each YOLO checkpoint:

```bash
python3 scripts/demo/run_demo.py \
  --weights_dir runs/detect --pattern best.pt \
  --run_two_stage \
  --classifier_weights runs/classifier_27k_finetuned/best.pt \
  --two_stage_split test
```

Outputs:
- Step 3 prediction JSON files in `runs/two_stage_baseline/` with suffix:
  `demo_<checkpoint-slug>`
- Step 4 printed metrics for each checkpoint/suffix pair

References:
- `scripts/two_stage_baseline/step3_two_stage_inference.py`
- `scripts/two_stage_baseline/step4_evaluate_two_stage.py`

### Output artifacts (for reporting)

- Detector summary CSV: `runs/demo/val_test_metrics_summary.csv`
- Detector per-model JSONs: `runs/demo/val_test_metrics__*.json`
- Two-stage predictions (if enabled): `runs/two_stage_baseline/predictions_*_demo_*.json`

### Troubleshooting

- **`Missing weights files`**: use real filesystem paths, not placeholders like `/path/to/...`.
- **No checkpoints found**: verify `--weights_dir` and `--pattern`.
- **Robustness skipped**: create corrupted sets first with `scripts/robustness/step1_create_corrupted_test_sets.py`.
- **Two-stage classifier missing**: pass a valid `--classifier_weights` path.

