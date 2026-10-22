"""
Converts COCO format annotations into YOLOv3 format.
COCO bounding box: [x, y, width, height] in absolute pixels.
YOLOv3 bounding box: [class_id, x_center/img_w, y_center/img_h, w/img_w, h/img_h] (0-indexed).
Also generates dataset/data.yaml for YOLO training and inference.
"""

import os
import json
from pathlib import Path
import yaml

import config

def coco_to_yolo_bbox(bbox, img_w, img_h):
    """
    Converts [x, y, w, h] in pixels to [x_center, y_center, w_norm, h_norm] normalized to [0, 1].
    Clamps values to [0.0, 1.0] to prevent out-of-bounds errors.
    """
    x, y, w, h = bbox
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return xc, yc, wn, hn

def convert_coco_split_to_yolo(coco_json_path, labels_out_dir, target_classes=config.TARGET_CLASSES):
    """
    Converts a COCO split JSON file into YOLO label .txt files.
    Note: In COCO subset, classes are 1..6. In YOLO, classes are 0..5 (0-indexed).
    """
    coco_json_path = Path(coco_json_path)
    labels_out_dir = Path(labels_out_dir)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in data["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    converted_count = 0
    for img in data["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        stem = Path(file_name).stem
        txt_path = labels_out_dir / f"{stem}.txt"

        img_w = float(img["width"])
        img_h = float(img["height"])
        if img_w <= 0 or img_h <= 0:
            continue

        anns = img_id_to_anns.get(img_id, [])
        lines = []
        for ann in anns:
            # COCO class_id is 1-indexed (1 to 6) -> YOLO is 0-indexed (0 to 5)
            yolo_cls_id = int(ann["category_id"]) - 1
            if yolo_cls_id < 0 or yolo_cls_id >= len(target_classes):
                continue
            
            bbox = ann.get("bbox", [])
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue

            xc, yc, wn, hn = coco_to_yolo_bbox(bbox, img_w, img_h)
            lines.append(f"{yolo_cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(lines) + "\n")
        converted_count += 1

    print(f"[YOLO Converter] Converted {converted_count} image annotations to YOLO format in {labels_out_dir}")
    return converted_count

def generate_yolo_data_yaml(
    data_yaml_path=config.YOLO_DATA_YAML,
    dataset_dir=config.DATASET_DIR,
    target_classes=config.TARGET_CLASSES
):
    """Generates the data.yaml file required for YOLO training and evaluation."""
    data_yaml_path = Path(data_yaml_path)
    yaml_dict = {
        "path": str(dataset_dir.as_posix()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "nc": len(target_classes),
        "names": target_classes
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"[YOLO Converter] Generated YOLO dataset config at: {data_yaml_path}")
    return data_yaml_path
