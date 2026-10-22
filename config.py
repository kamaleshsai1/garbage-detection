"""
Configuration module for Garbage Detection and Classification Using Faster R-CNN and YOLOv3.
Author: Antigravity Project Team
Dataset: TACO (Trash Annotations in Context)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"
SUBSET_COCO_FILE = DATASET_DIR / "taco_subset_coco.json"
SCALED_GT_FILE = DATASET_DIR / "taco_annotations_scaled.json"
YOLO_DATA_YAML = DATASET_DIR / "data.yaml"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
DOCS_DIR = BASE_DIR / "docs"

for directory in [DATASET_DIR, IMAGES_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, 
                  EVALUATIONS_DIR, FIGURES_DIR, SAMPLES_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

TARGET_CLASSES = [
    "Bottle",
    "Can",
    "Plastic Wrapper",
    "Carton",
    "Cup",
    "Other Litter"
]

CLASS_COLORS = {
    "Bottle": (0, 165, 255),
    "Can": (255, 105, 180),
    "Plastic Wrapper": (50, 205, 50),
    "Carton": (218, 165, 32),
    "Cup": (255, 140, 0),
    "Other Litter": (147, 112, 219)
}

CLASS_HEX_COLORS = {
    "Bottle": "#FFA500",
    "Can": "#FF69B4",
    "Plastic Wrapper": "#32CD32",
    "Carton": "#DAA520",
    "Cup": "#FF8C00",
    "Other Litter": "#9370DB"
}

SUPER_CATEGORY_MAP = {
    "Bottle": "Bottle",
    "Can": "Can",
    "Plastic bag & wrapper": "Plastic Wrapper",
    "Carton": "Carton",
    "Paper": "Carton",
    "Cup": "Cup",
    "Bottle cap": "Other Litter",
    "Cigarette": "Other Litter",
    "Straw": "Other Litter",
    "Broken glass": "Other Litter",
    "Styrofoam piece": "Other Litter",
    "Pop tab": "Other Litter",
    "Lid": "Other Litter",
    "Aluminium foil": "Other Litter",
    "Blister pack": "Other Litter",
    "Other plastic": "Plastic Wrapper"
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

FUSION_CONFIG = {
    "iou_threshold": 0.30,
    "faster_rcnn_conf_thresh": 0.20,
    "yolov3_conf_thresh": 0.20,
    "fused_conf_thresh": 0.20,
    "box_fusion_method": "weighted_average",
    "bonus_factor": 1.20
}

FASTER_RCNN_WEIGHTS = CHECKPOINTS_DIR / "faster_rcnn_garbage_best.pth"
YOLOV3_WEIGHTS = CHECKPOINTS_DIR / "yolov3_garbage_best.pth"
COCO_PRETRAINED_YOLOV3 = CHECKPOINTS_DIR / "yolov3_coco.pth"

TRAIN_CONFIG = {
    "batch_size": 4 if DEVICE == "cuda" else 2,
    "faster_rcnn_lr": 0.001,
    "faster_rcnn_momentum": 0.9,
    "faster_rcnn_weight_decay": 0.0005,
    "yolov3_lr": 0.001,
    "num_epochs": 10,
    "num_workers": 0,
    "img_size": 416
}

YOLO_IMG_SIZE = 416
