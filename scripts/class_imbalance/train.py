"""
Train YOLOv11 on NIH malaria dataset.
Uses config for reproducibility. Run create_splits.py and convert_to_yolo.py first.
Supports class_weights from config for imbalanced data (parasitized vs uninfected).

Source / references:
- Ultralytics YOLO API, DetectionTrainer, v8DetectionLoss: https://github.com/ultralytics/ultralytics
- Custom WeightedDetectionLoss and MalariaDetectionTrainer extend ultralytics (same repo).
"""

import argparse  # lets the user run the script with flags like --oversample or --resume
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO  # Ultralytics YOLO class (loads model + runs train/val)
from ultralytics.models.yolo.detect import DetectionTrainer  # Ultralytics training loop for detection
from ultralytics.utils import DEFAULT_CFG # default Ultralytics training config object
from ultralytics.utils.loss import v8DetectionLoss  # default YOLOv8/11 detection loss used by Ultralytics
from ultralytics.utils.tal import make_anchors 

# Reproducibility
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DATASET_PATH = PROJECT_ROOT / "config" / "dataset.yaml"
DATASET_OVERSAMPLED_PATH = PROJECT_ROOT / "config" / "dataset_oversampled.yaml"


class WeightedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with per-class weighting of cls loss (for imbalanced data)."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs) # initialise the standard Ultralytics detection loss
        self._model_ref = model  # store the model so the loss can read model.class_weights later

    def get_assigned_targets_and_loss(self, preds, batch):
        """Same as parent but weight cls loss by model.class_weights if set."""
        # The code first checks if class weights exist on the model.
        # If they don’t exist, it behaves exactly like the original Ultralytics loss.
        weights = getattr(self._model_ref, "class_weights", None)
        if weights is None:
            return super().get_assigned_targets_and_loss(preds, batch)

        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # Weighted cls loss: weight each class by class_weights
        bce = self.bce(pred_scores, target_scores.to(dtype))
        w = weights.to(pred_scores.device).to(dtype).view(1, 1, -1)
        loss[1] = (bce * w).sum() / (target_scores.to(dtype) * w).sum().clamp(min=1)

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class MalariaDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that sets model.class_weights and uses WeightedDetectionLoss."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self._class_weights = overrides.pop("class_weights", None)
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        super().set_model_attributes()
        weights = getattr(self, "_class_weights", None)
        if weights is not None and len(weights) == self.data["nc"]:
            device = next(self.model.parameters()).device
            self.model.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            # Use weighted loss (don't replace init_criterion: lambda is not picklable on save)
            self.model.criterion = WeightedDetectionLoss(self.model)


def train():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on malaria dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in project/name")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation 0-1 (0 disables; can avoid crash)")
    parser.add_argument("--oversample", action="store_true", help="Condition C: use oversampled train list (parasitized images 3×). Run scripts/class_imbalance/build_oversampled_train_list.py first.")
    parser.add_argument("--weighted", action="store_true", help="With --oversample: Condition D (oversample + class weights). Else ignored.")
    args, _ = parser.parse_known_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    tr = config["training"]
    # Condition C: oversample only. Condition D: oversample + weighted.
    if args.oversample:
        data_path = DATASET_OVERSAMPLED_PATH
        if not data_path.exists():
            raise FileNotFoundError(f"Oversampled dataset config not found: {data_path}. Run: python3 scripts/class_imbalance/build_oversampled_train_list.py")
        overrides_name = "malaria_oversampled_weighted" if args.weighted else "malaria_oversampled"
    else:
        data_path = DATASET_PATH
        overrides_name = tr["name"]

    overrides = {
        "data": str(data_path),
        "epochs": tr["epochs"],
        "batch": args.batch if args.batch is not None else tr["batch"],
        "imgsz": tr["imgsz"],
        "patience": tr["patience"],
        "save": tr["save"],
        "project": str(PROJECT_ROOT / tr["project"]),
        "name": overrides_name,
        "seed": config["seed"],
    }
    if args.mosaic is not None:
        overrides["mosaic"] = args.mosaic
    if args.resume:
        overrides["resume"] = True
        # Use overrides_name (e.g. malaria_oversampled_weighted for D) not tr["name"]
        ckpt = PROJECT_ROOT / tr["project"] / overrides_name / "weights" / "last.pt"
        if ckpt.exists():
            model = YOLO(str(ckpt))
            print(f"Resuming from {ckpt}")
        else:
            model = YOLO(tr["model"])
            print("No last.pt found; starting from scratch (resume=True will take effect after first run).")
    else:
        model = YOLO(tr["model"])

    # Condition C: oversample, no weights. Condition D: oversample + weights.
    class_weights = config.get("class_weights") if (not args.oversample or args.weighted) else None
    if class_weights is not None:
        overrides["class_weights"] = class_weights
        trainer_cls = MalariaDetectionTrainer
    else:
        trainer_cls = None

    model.train(trainer=trainer_cls, **overrides)


if __name__ == "__main__":
    train()
