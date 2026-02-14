#!/bin/bash
# Run from project root. Executes the granular commit plan.
set -e
cd "$(dirname "$0")/.."

echo "=== Commit 1: config/default.yaml ==="
git add config/default.yaml && git commit -m "Add class_weights to config for imbalanced parasitized/uninfected" || true

echo "=== Commit 2: scripts/compute_class_weights.py ==="
git add scripts/compute_class_weights.py && git commit -m "Add compute_class_weights.py for inverse-frequency weighting" || true

echo "=== Commit 3: scripts/train.py ==="
git add scripts/train.py && git commit -m "Add WeightedDetectionLoss, --oversample, --weighted for Conditions B/C/D" || true

echo "=== Commit 4: config/dataset_oversampled.yaml ==="
git add config/dataset_oversampled.yaml && git commit -m "Add dataset_oversampled.yaml for Conditions C and D" || true

echo "=== Commit 5: scripts/build_oversampled_train_list.py ==="
git add scripts/build_oversampled_train_list.py && git commit -m "Add build_oversampled_train_list.py for training-time oversampling" || true

echo "=== Commit 6: scripts/evaluate_conditions.py ==="
git add scripts/evaluate_conditions.py && git commit -m "Add evaluate_conditions.py for A/B/C/D comparison table" || true

echo "=== Commit 7: config/dataset.yaml ==="
git add config/dataset.yaml && git commit -m "Update dataset.yaml: add source comment" || true

echo "=== Commit 8: scripts/convert_to_yolo.py ==="
git add scripts/convert_to_yolo.py && git commit -m "Update convert_to_yolo.py" || true

echo "=== Commit 9: scripts/create_splits.py ==="
git add scripts/create_splits.py && git commit -m "Update create_splits.py" || true

echo "=== Commit 10: scripts/verify_conversion.py ==="
git add scripts/verify_conversion.py && git commit -m "Update verify_conversion.py" || true

echo "=== Commits 11-13: data/splits (if untracked/modified) ==="
git add data/splits/train_patients.csv && git commit -m "Add train patient split" || true
git add data/splits/val_patients.csv && git commit -m "Add val patient split" || true
git add data/splits/test_patients.csv && git commit -m "Add test patient split" || true

echo "=== Commit 14: requirements.txt ==="
git add requirements.txt && git commit -m "Add requirements.txt" || true

echo "=== Commit 15: README.md ==="
git add README.md && git commit -m "Update README" || true

echo "=== Commit 16: scripts/commit_plan.md ==="
git add scripts/commit_plan.md && git commit -m "Add commit plan for granular history" || true

echo "=== Commit 17: scripts/run_commit_plan.sh ==="
git add scripts/run_commit_plan.sh && git commit -m "Add run_commit_plan.sh" || true

echo "=== Done. Run: git log --oneline ==="
