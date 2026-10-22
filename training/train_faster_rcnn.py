"""
Training and Fine-Tuning Pipeline for Faster R-CNN on TACO Garbage Dataset.
Configurable for CPU and GPU environments.
Includes data loading, learning rate scheduling, validation monitoring,
and model checkpoint saving.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.faster_rcnn_detector import get_faster_rcnn_model

class TACOFasterRCNNDataset(Dataset):
    """
    COCO-format dataset loader for Faster R-CNN.
    Returns:
        image: float tensor (3, H, W) normalized to [0, 1]
        target: dict with 'boxes' (N, 4), 'labels' (N,), 'image_id' (1,)
    """
    def __init__(self, coco_json_path, images_dir=config.IMAGES_DIR, target_classes=config.TARGET_CLASSES):
        self.images_dir = Path(images_dir)
        self.target_classes = target_classes

        with open(coco_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]
        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in data["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Filter out images without existing file on disk if images have been downloaded
        self.valid_images = []
        for img in self.images:
            img_path = self.images_dir / img["file_name"]
            if img_path.exists():
                self.valid_images.append(img)
            else:
                # Also check flat basename in images_dir
                flat_path = self.images_dir / Path(img["file_name"]).name
                if flat_path.exists():
                    img_copy = dict(img)
                    img_copy["file_name"] = Path(img["file_name"]).name
                    self.valid_images.append(img_copy)

        print(f"[Faster R-CNN Dataset] Found {len(self.valid_images)} accessible images from {len(self.images)} total entries.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_info = self.valid_images[idx]
        img_id = img_info["id"]
        img_path = self.images_dir / img_info["file_name"]
        
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        raw_anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []

        for ann in raw_anns:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, bw, bh = bbox
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(w), x1 + float(bw))
            y2 = min(float(h), y1 + float(bh))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                # COCO category_id (1..6)
                labels.append(int(ann["category_id"]))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        img_tensor = F.to_tensor(image)
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_faster_rcnn(
    train_json=config.DATASET_DIR / "train_coco.json",
    val_json=config.DATASET_DIR / "val_coco.json",
    epochs=config.TRAIN_CONFIG["num_epochs"],
    batch_size=config.TRAIN_CONFIG["batch_size"],
    lr=config.TRAIN_CONFIG["faster_rcnn_lr"],
    device=config.DEVICE,
    save_path=config.FASTER_RCNN_WEIGHTS
):
    """
    Executes training loop for Faster R-CNN.
    """
    device = torch.device(device)
    print(f"[Faster R-CNN Training] Starting training on device: {device}")

    # Datasets and Loaders
    train_dataset = TACOFasterRCNNDataset(train_json)
    if len(train_dataset) == 0:
        print("[Faster R-CNN Training] No local training images found. Please run preprocessing/download first.")
        return None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.TRAIN_CONFIG["num_workers"]
    )

    val_dataset = TACOFasterRCNNDataset(val_json)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.TRAIN_CONFIG["num_workers"]
    ) if len(val_dataset) > 0 else None

    # Model initialization
    num_classes = len(config.TARGET_CLASSES) + 1
    model = get_faster_rcnn_model(num_classes=num_classes, pretrained_backbone=True)
    model.to(device)

    # Optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=config.TRAIN_CONFIG["faster_rcnn_momentum"],
        weight_decay=config.TRAIN_CONFIG["faster_rcnn_weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    training_history = {
        "epochs": [],
        "train_loss": [],
        "loss_classifier": [],
        "loss_box_reg": [],
        "loss_objectness": [],
        "loss_rpn_box_reg": []
    }

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        cls_losses = []
        box_losses = []
        obj_losses = []
        rpn_box_losses = []

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            epoch_losses.append(losses.item())
            cls_losses.append(loss_dict.get("loss_classifier", torch.tensor(0.0)).item())
            box_losses.append(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item())
            obj_losses.append(loss_dict.get("loss_objectness", torch.tensor(0.0)).item())
            rpn_box_losses.append(loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item())

        lr_scheduler.step()

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        mean_cls = float(np.mean(cls_losses)) if cls_losses else 0.0
        mean_box = float(np.mean(box_losses)) if box_losses else 0.0
        mean_obj = float(np.mean(obj_losses)) if obj_losses else 0.0
        mean_rpn = float(np.mean(rpn_box_losses)) if rpn_box_losses else 0.0

        training_history["epochs"].append(epoch)
        training_history["train_loss"].append(round(mean_loss, 4))
        training_history["loss_classifier"].append(round(mean_cls, 4))
        training_history["loss_box_reg"].append(round(mean_box, 4))
        training_history["loss_objectness"].append(round(mean_obj, 4))
        training_history["loss_rpn_box_reg"].append(round(mean_rpn, 4))

        print(f"Epoch [{epoch}/{epochs}] - Total Loss: {mean_loss:.4f} (Cls: {mean_cls:.4f}, Box: {mean_box:.4f}, Obj: {mean_obj:.4f})")

        # Save best weights
        if mean_loss < best_loss:
            best_loss = mean_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "classes": config.TARGET_CLASSES
            }, save_path)
            print(f"  --> Best model checkpoint saved to: {save_path}")

    # Save training log JSON
    log_file = config.EVALUATIONS_DIR / "faster_rcnn_training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    print(f"[Faster R-CNN Training] Training log saved to: {log_file}")

    return training_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=config.TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=config.TRAIN_CONFIG["faster_rcnn_lr"])
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    train_faster_rcnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
