"""
Leakage-Free Dataset Splitter.
Splits images into Train (70%), Val (15%), and Test (15%) splits
with a fixed random seed (RANDOM_SEED = 42).
Generates split COCO JSON files and split YOLO image lists.
"""

import os
import json
from pathlib import Path
import numpy as np

import config

def split_dataset(
    subset_coco_path=config.SUBSET_COCO_FILE,
    split_ratios=config.SPLIT_RATIOS,
    random_seed=config.RANDOM_SEED,
    output_dir=config.DATASET_DIR
):
    """
    Performs leakage-free splitting of the TACO dataset at the image level.
    Outputs:
      - train_coco.json, val_coco.json, test_coco.json
      - train.txt, val.txt, test.txt (lists of image file paths for YOLO)
    """
    output_dir = Path(output_dir)
    with open(subset_coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # Sort images by ID to guarantee deterministic permutation
    images_sorted = sorted(images, key=lambda x: x["id"])
    num_images = len(images_sorted)

    rng = np.random.RandomState(random_seed)
    shuffled_indices = rng.permutation(num_images)

    train_pct = split_ratios["train"]
    val_pct = split_ratios["val"]

    train_end = int(num_images * train_pct)
    val_end = train_end + int(num_images * val_pct)

    train_imgs = [images_sorted[i] for i in shuffled_indices[:train_end]]
    val_imgs = [images_sorted[i] for i in shuffled_indices[train_end:val_end]]
    test_imgs = [images_sorted[i] for i in shuffled_indices[val_end:]]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    # Map annotations to splits
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    summary = {}
    for split_name, split_image_list in splits.items():
        split_image_ids = {img["id"] for img in split_image_list}
        split_anns = []
        for img_id in split_image_ids:
            split_anns.extend(ann_by_image.get(img_id, []))

        split_coco = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "images": split_image_list,
            "annotations": split_anns,
            "categories": categories
        }

        # Save COCO JSON
        json_path = output_dir / f"{split_name}_coco.json"
        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(split_coco, f_out, indent=2)

        # Save YOLO image list
        txt_path = output_dir / f"{split_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            for img in split_image_list:
                img_path = (config.IMAGES_DIR / img["file_name"]).as_posix()
                f_txt.write(img_path + "\n")

        summary[split_name] = {
            "num_images": len(split_image_list),
            "num_annotations": len(split_anns),
            "json_path": str(json_path),
            "txt_path": str(txt_path)
        }

    # Verify zero leakage
    train_ids = {img["id"] for img in splits["train"]}
    val_ids = {img["id"] for img in splits["val"]}
    test_ids = {img["id"] for img in splits["test"]}
    
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage between val and test!"

    print("[Dataset Splitter] Leakage-free dataset partition created:")
    print(f"  Train: {summary['train']['num_images']} images, {summary['train']['num_annotations']} annotations")
    print(f"  Val:   {summary['val']['num_images']} images, {summary['val']['num_annotations']} annotations")
    print(f"  Test:  {summary['test']['num_images']} images, {summary['test']['num_annotations']} annotations")
    print("  Zero image overlap verified across all splits.")

    return summary
