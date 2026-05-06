## Supervisor demo (run with `best.pt`)

This folder provides a minimal, copy-paste demo to run evaluation scripts against a YOLO checkpoint
(`best.pt`) using the same repository conventions.

### Prerequisites

- Python dependencies installed:

```bash
pip install -r requirements.txt
```

- Dataset present under `data/processed/` (images + labels). See repo `README.md` for dataset setup.

### 1) Quick test: run Ultralytics validation on the test split

```bash
python3 scripts/demo/run_demo.py --weights /path/to/best.pt
```

Outputs:
- Prints key metrics to stdout.
- Writes a small JSON summary under `runs/demo/`.

### 2) Robustness (optional)

If `data/processed_corrupted/` exists (created by `scripts/robustness/step1_create_corrupted_test_sets.py`),
the demo can run the YOLO robustness Step 2 script with the provided weights:

```bash
python3 scripts/demo/run_demo.py --weights /path/to/best.pt --run_robustness
```

References:
- `scripts/robustness/step1_create_corrupted_test_sets.py`
- `scripts/robustness/step2_run_yolo_robustness.py`

