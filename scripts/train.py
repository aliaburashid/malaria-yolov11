"""
Train YOLOv11 on NIH malaria dataset.
Uses config for reproducibility. Run create_splits.py and convert_to_yolo.py first.
"""

import yaml
from pathlib import Path
from ultralytics import YOLO

# Reproducibility
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DATASET_PATH = PROJECT_ROOT / "config" / "dataset.yaml"


def train():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    tr = config["training"]
    model = YOLO(tr["model"])

    model.train(
        data=str(DATASET_PATH),
        epochs=tr["epochs"],
        batch=tr["batch"],
        imgsz=tr["imgsz"],
        patience=tr["patience"],
        save=tr["save"],
        project=str(PROJECT_ROOT / tr["project"]),
        name=tr["name"],
        seed=config["seed"],
    )


if __name__ == "__main__":
    train()
