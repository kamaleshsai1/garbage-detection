"""
TACO Dataset Loader and Annotation Preprocessor.
Handles downloading annotations, image fetching, class consolidation into
6 target classes, and generating filtered COCO annotations.
"""

import os
import json
import urllib.request
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import requests

import config

def download_taco_annotations(dest_path=config.ANNOTATIONS_FILE, overwrite=False):
    """Downloads the official TACO annotations.json from GitHub if not present."""
    dest_path = Path(dest_path)
    if dest_path.exists() and not overwrite:
        print(f"[TACO Loader] Annotations file exists at: {dest_path}")
        return dest_path

    url = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
    print(f"[TACO Loader] Downloading official annotations from: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
        out.write(resp.read())
    
    print(f"[TACO Loader] Saved annotations to: {dest_path}")
    return dest_path

def inspect_dataset(annotations_path=config.ANNOTATIONS_FILE):
    """Reads and returns statistics on images, annotations, categories, and supercategories."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = {c["id"]: c for c in data.get("categories", [])}

    supercat_counts = Counter()
    cat_counts = Counter()
    
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_counts[cat_id] += 1
        if cat_id in categories:
            supercat_counts[categories[cat_id]["supercategory"]] += 1

    stats = {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "cat_counts": dict(cat_counts),
        "supercat_counts": dict(supercat_counts),
        "categories": categories
    }
    return stats

def map_taco_category_to_target(taco_cat, supercat_map=config.SUPER_CATEGORY_MAP):
    """
    Maps an individual TACO category (with name and supercategory)
    to one of the 6 target classes.
    """
    scat = taco_cat.get("supercategory", "")
    name = taco_cat.get("name", "").lower()
    
    # Direct supercategory mapping
    if scat in supercat_map:
        return supercat_map[scat]
    
    # Name-based heuristic fallbacks
    if "bottle" in name:
        return "Bottle"
    elif "can" in name:
        return "Can"
    elif any(kw in name for kw in ["plastic", "wrapper", "bag", "film", "packet"]):
        return "Plastic Wrapper"
    elif any(kw in name for kw in ["carton", "box", "cardboard", "paper"]):
        return "Carton"
    elif "cup" in name:
        return "Cup"
    else:
        return "Other Litter"

def build_subset_coco_dataset(
    annotations_path=config.ANNOTATIONS_FILE,
    output_path=config.SUBSET_COCO_FILE,
    target_classes=config.TARGET_CLASSES,
    min_box_size=10
):
    """
    Processes official TACO annotations, maps categories into target classes,
    filters small invalid bounding boxes, and generates taco_subset_coco.json.
    Faster R-CNN category IDs: 1 to len(target_classes) (0 is background).
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    taco_cats = {c["id"]: c for c in raw_data["categories"]}
    
    # Create target categories
    # Note: COCO format starts target categories at ID 1 (ID 0 is reserved for background in detection models)
    target_cat_to_id = {cls_name: i + 1 for i, cls_name in enumerate(target_classes)}
    new_categories = [
        {"id": target_cat_to_id[cls_name], "name": cls_name, "supercategory": "Garbage"}
        for cls_name in target_classes
    ]

    # Map raw annotation category IDs to new category IDs
    raw_cat_to_target_id = {}
    for raw_id, c_info in taco_cats.items():
        mapped_target = map_taco_category_to_target(c_info)
        if mapped_target in target_cat_to_id:
            raw_cat_to_target_id[raw_id] = target_cat_to_id[mapped_target]

    filtered_annotations = []
    referenced_image_ids = set()

    for ann in raw_data["annotations"]:
        raw_cid = ann["category_id"]
        if raw_cid not in raw_cat_to_target_id:
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if w < min_box_size or h < min_box_size:
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = raw_cat_to_target_id[raw_cid]
        filtered_annotations.append(new_ann)
        referenced_image_ids.add(ann["image_id"])

    # Retain only images with valid annotations
    filtered_images = [
        img for img in raw_data["images"] if img["id"] in referenced_image_ids
    ]

    subset_data = {
        "info": raw_data.get("info", {}),
        "licenses": raw_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": new_categories
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, indent=2)

    print(f"[TACO Loader] Generated subset COCO dataset at: {output_path}")
    print(f"  Valid images: {len(filtered_images)}")
    print(f"  Valid annotations: {len(filtered_annotations)}")
    
    # Class breakdown
    counts = Counter(a["category_id"] for a in filtered_annotations)
    for cls_name in target_classes:
        cid = target_cat_to_id[cls_name]
        print(f"  - Class '{cls_name}' (ID {cid}): {counts[cid]} annotations")

    return subset_data

def download_taco_images(
    subset_coco_path=config.SUBSET_COCO_FILE,
    output_dir=config.IMAGES_DIR,
    max_images=100,
    use_resized=True
):
    """
    Downloads images listed in subset_coco_path to output_dir.
    Handles network errors, broken URLs, and existing files gracefully.
    """
    with open(subset_coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"][:max_images]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TACO Loader] Preparing to download up to {len(images)} images to {output_dir}...")
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for img_info in tqdm(images, desc="Downloading TACO Images"):
        file_name = img_info["file_name"]
        # Maintain batch subdirectories or flat structure
        save_path = output_dir / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists() and save_path.stat().st_size > 1000:
            skipped_count += 1
            success_count += 1
            continue

        url = img_info.get("flickr_640_url") if use_resized else img_info.get("flickr_url")
        if not url:
            url = img_info.get("flickr_url")

        if not url:
            failed_count += 1
            continue

        try:
            resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                with open(save_path, "wb") as f_out:
                    f_out.write(resp.content)
                success_count += 1
            else:
                failed_count += 1
        except Exception:
            failed_count += 1

    print(f"[TACO Loader] Download complete: {success_count} ready ({skipped_count} cached), {failed_count} failed/skipped.")
    return success_count
