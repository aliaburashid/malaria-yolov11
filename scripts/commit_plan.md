# Commit Plan: Granular Commit Structure

Run these commands in order from the project root. Each commit is self-contained.

## Core config and scripts (11 commits)

### 1. Config: add class_weights
```bash
git add config/default.yaml
git commit -m "Add class_weights to config for imbalanced parasitized/uninfected"
```

### 2. Script: compute_class_weights.py
```bash
git add scripts/compute_class_weights.py
git commit -m "Add compute_class_weights.py for inverse-frequency weighting"
```

### 3. Script: train.py
```bash
git add scripts/train.py
git commit -m "Add WeightedDetectionLoss, --oversample, --weighted for Conditions B/C/D"
```

### 4. Config: dataset_oversampled.yaml
```bash
git add config/dataset_oversampled.yaml
git commit -m "Add dataset_oversampled.yaml for Conditions C and D"
```

### 5. Script: build_oversampled_train_list.py
```bash
git add scripts/build_oversampled_train_list.py
git commit -m "Add build_oversampled_train_list.py for training-time oversampling"
```

### 6. Script: evaluate_conditions.py
```bash
git add scripts/evaluate_conditions.py
git commit -m "Add evaluate_conditions.py for A/B/C/D comparison table"
```

### 7. Config: dataset.yaml
```bash
git add config/dataset.yaml
git commit -m "Update dataset.yaml: add source comment"
```

### 8. Script: convert_to_yolo.py
```bash
git add scripts/convert_to_yolo.py
git commit -m "Update convert_to_yolo.py"
```

### 9. Script: create_splits.py
```bash
git add scripts/create_splits.py
git commit -m "Update create_splits.py"
```

### 10. Script: verify_conversion.py
```bash
git add scripts/verify_conversion.py
git commit -m "Update verify_conversion.py"
```

### 11. Data splits (if modified/untracked)
```bash
git add data/splits/train_patients.csv
git commit -m "Add train patient split" || true
git add data/splits/val_patients.csv
git commit -m "Add val patient split" || true
git add data/splits/test_patients.csv
git commit -m "Add test patient split" || true
```

## Extra commits (run if files exist and have changes)

### 12. requirements.txt
```bash
git add requirements.txt
git commit -m "Add requirements.txt" || true
```

### 13. README.md
```bash
git add README.md
git commit -m "Update README" || true
```

### 14. Commit plan (this file)
```bash
git add scripts/commit_plan.md
git commit -m "Add commit plan for granular history"
```

### 15. Run script
```bash
git add scripts/run_commit_plan.sh
git commit -m "Add run_commit_plan.sh"
```

## Even more commits (use `git add -p` to split)

To split a single file into multiple commits, use patch mode:

```bash
# Example: split config/default.yaml into 2 commits
git add -p config/default.yaml   # Stage only class_weights hunk, commit
git add config/default.yaml      # Stage remainder, commit
```

You can similarly split `train.py` (WeightedDetectionLoss, MalariaDetectionTrainer, CLI flags) or `evaluate_conditions.py` (run_val, table, CSV) for 2–3 extra commits per file.

## Run-all script
```bash
chmod +x scripts/run_commit_plan.sh
./scripts/run_commit_plan.sh
```

## Notes
- Use `|| true` so a step is skipped if there's nothing to commit
- Run `git status` between steps to see what's left
