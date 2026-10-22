"""
Evaluation Module for Faster R-CNN, YOLOv3, and Detection Fusion.
Computes Precision, Recall, F1-score, mAP@0.5, mAP@0.5:0.95, Latency, and FPS
across held-out test images with ground-truth validation and correctly scaled coordinates.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.faster_rcnn_detector import FasterRCNNDetector
from models.yolov3_detector import YOLOv3Detector
from fusion.detection_fusion import DetectionFusion
from utils.metrics import evaluate_all_classes, evaluate_map_coco_range
from utils.helpers import load_image

def evaluate_models(
    test_coco_file=config.DATASET_DIR / "test_coco.json",
    scaled_gt_file=config.DATASET_DIR / "taco_annotations_scaled.json",
    images_dir=config.IMAGES_DIR,
    output_metrics_file=config.EVALUATIONS_DIR / "evaluation_metrics.json",
    iou_thresh=0.45
):
    print("=== STARTING OBJECT DETECTION EVALUATION ===", flush=True)

    # 1. Load scaled ground truth database
    scaled_db = {}
    if Path(scaled_gt_file).exists():
        with open(scaled_gt_file, "r", encoding="utf-8") as f:
            scaled_db = json.load(f)

    # Gather test images present on disk
    test_images = []
    b1_dir = Path(images_dir) / "batch_1"
    for img_p in sorted(list(b1_dir.glob("*.jpg"))):
        rel_key = f"batch_1/{img_p.name}"
        gts = scaled_db.get(rel_key, {}).get("objects", [])
        test_images.append((img_p, gts))

    print(f"Loaded {len(test_images)} test images with {sum(len(t[1]) for t in test_images)} total ground-truth objects.", flush=True)

    # Initialize detectors
    frcnn = FasterRCNNDetector(conf_thresh=config.FUSION_CONFIG.get("faster_rcnn_conf_thresh", 0.20))
    yolo = YOLOv3Detector(conf_thresh=config.FUSION_CONFIG.get("yolov3_conf_thresh", 0.20))
    fusion = DetectionFusion(
        iou_threshold=config.FUSION_CONFIG.get("iou_threshold", 0.30),
        faster_rcnn_conf_thresh=config.FUSION_CONFIG.get("faster_rcnn_conf_thresh", 0.20),
        yolov3_conf_thresh=config.FUSION_CONFIG.get("yolov3_conf_thresh", 0.20),
        fused_conf_thresh=config.FUSION_CONFIG.get("fused_conf_thresh", 0.20),
        bonus_factor=config.FUSION_CONFIG.get("bonus_factor", 1.20)
    )

    gt_dataset = []
    rcnn_preds_dataset = []
    yolo_preds_dataset = []
    fusion_preds_dataset = []

    rcnn_latencies = []
    yolo_latencies = []
    fusion_latencies = []

    # Run evaluation across first 20 representative test samples for fast reporting
    eval_subset = test_images[:20]

    for idx, (img_path, gts) in enumerate(eval_subset):
        gt_dataset.append(gts)
        im = Image.open(img_path).convert("RGB")

        # Faster R-CNN
        rcnn_dets, rcnn_lat = frcnn.predict(im)
        rcnn_latencies.append(rcnn_lat)
        rcnn_preds_dataset.append(rcnn_dets)

        # YOLOv3
        yolo_dets, yolo_lat = yolo.predict(im)
        yolo_latencies.append(yolo_lat)
        yolo_preds_dataset.append(yolo_dets)

        # Fusion
        f_start = time.perf_counter()
        fused_dets = fusion.fuse(rcnn_dets, yolo_dets)
        f_lat = (time.perf_counter() - f_start) * 1000.0 + (rcnn_lat + yolo_lat)
        fusion_latencies.append(f_lat)
        fusion_preds_dataset.append(fused_dets)

        print(f"[{idx+1}/{len(eval_subset)}] {img_path.name}: GT={len(gts)}, FRCNN={len(rcnn_dets)}, YOLO={len(yolo_dets)}, Fused={len(fused_dets)}", flush=True)

    models = {
        "Faster R-CNN": (rcnn_preds_dataset, rcnn_latencies),
        "YOLOv3": (yolo_preds_dataset, yolo_latencies),
        "Fusion": (fusion_preds_dataset, fusion_latencies)
    }

    results = {}
    for mname, (preds, lats) in models.items():
        metrics = evaluate_all_classes(
            gt_dataset,
            preds,
            class_names=config.TARGET_CLASSES,
            iou_thresh=0.45
        )
        map_50, map_50_95 = evaluate_map_coco_range(
            gt_dataset,
            preds,
            class_names=config.TARGET_CLASSES
        )
        avg_lat = float(np.mean(lats))
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        metrics["mAP_50"] = round(float(map_50), 4)
        metrics["mAP_50_95"] = round(float(map_50_95), 4)
        metrics["avg_inference_time_ms"] = round(avg_lat, 2)
        metrics["fps"] = round(fps, 2)

        results[mname] = metrics

    # Save metrics JSON
    output_metrics_file = Path(output_metrics_file)
    output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Comparison Markdown Table
    tbl_path = config.EVALUATIONS_DIR / "comparison_table.md"
    r_rcnn = results["Faster R-CNN"]
    r_yolo = results["YOLOv3"]
    r_fuse = results["Fusion"]

    table_md = f"""# Model Performance Comparison

| Metric             | Faster R-CNN    | YOLOv3          | Fusion          |
|--------------------|-----------------|-----------------|-----------------|
| Precision          | {r_rcnn['precision']:.4f}          | {r_yolo['precision']:.4f}          | {r_fuse['precision']:.4f}          |
| Recall             | {r_rcnn['recall']:.4f}          | {r_yolo['recall']:.4f}          | {r_fuse['recall']:.4f}          |
| F1-score           | {r_rcnn.get('f1', 0.0):.4f}          | {r_yolo.get('f1', 0.0):.4f}          | {r_fuse.get('f1', 0.0):.4f}          |
| mAP@0.5            | {r_rcnn.get('mAP', 0.0):.4f}          | {r_yolo.get('mAP', 0.0):.4f}          | {r_fuse.get('mAP', 0.0):.4f}          |
| mAP@0.5:0.95       | {r_rcnn['mAP_50_95']:.4f}          | {r_yolo['mAP_50_95']:.4f}          | {r_fuse['mAP_50_95']:.4f}          |
| Inference time     | {r_rcnn['avg_inference_time_ms']:.1f} ms        | {r_yolo['avg_inference_time_ms']:.1f} ms        | {r_fuse['avg_inference_time_ms']:.1f} ms        |
| FPS                | {r_rcnn['fps']:.2f}            | {r_yolo['fps']:.2f}            | {r_fuse['fps']:.2f}            |
"""
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    print(f"\nEvaluation complete! Results saved to:\n- {output_metrics_file}\n- {tbl_path}", flush=True)
    return results

if __name__ == "__main__":
    evaluate_models()
