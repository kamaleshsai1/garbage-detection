"""
Detection Fusion Algorithm for Faster R-CNN and YOLOv3.
Combines candidate detections using:
1. Common coordinate normalization [x1, y1, x2, y2]
2. Class-aware IoU matching with configurable IoU threshold (default 0.50)
3. Confidence-weighted bounding-box coordinate averaging
4. Calibrated fused confidence score with agreement bonus
5. Unmatched detection retention
6. Baseline comparisons (Weighted Box Fusion and NMS)
"""

import numpy as np
import config
from utils.metrics import calculate_iou

class DetectionFusion:
    def __init__(
        self,
        iou_threshold=config.FUSION_CONFIG["iou_threshold"],
        faster_rcnn_conf_thresh=config.FUSION_CONFIG["faster_rcnn_conf_thresh"],
        yolov3_conf_thresh=config.FUSION_CONFIG["yolov3_conf_thresh"],
        fused_conf_thresh=config.FUSION_CONFIG["fused_conf_thresh"],
        bonus_factor=config.FUSION_CONFIG["bonus_factor"]
    ):
        self.iou_threshold = float(iou_threshold)
        self.faster_rcnn_conf_thresh = float(faster_rcnn_conf_thresh)
        self.yolov3_conf_thresh = float(yolov3_conf_thresh)
        self.fused_conf_thresh = float(fused_conf_thresh)
        self.bonus_factor = float(bonus_factor)

    def fuse(self, faster_rcnn_dets, yolov3_dets):
        """
        Executes the detection fusion algorithm matching the paper methodology:
        Algorithm: Faster R-CNN + YOLOv3 Detection Fusion
        Input: RCNN_Detections, YOLO_Detections
        Output: Fused_Detections
        """
        # Step 1: Pre-filter by model-specific confidence thresholds
        rcnn_cands = [
            d for d in faster_rcnn_dets if d["score"] >= self.faster_rcnn_conf_thresh
        ]
        yolo_cands = [
            d for d in yolov3_dets if d["score"] >= self.yolov3_conf_thresh
        ]

        fused_detections = []
        matched_yolo_indices = set()

        # Step 2: For each detection R in RCNN_Detections
        for r_det in rcnn_cands:
            r_box = r_det["box"]
            r_class = r_det["class_name"]
            r_score = r_det["score"]

            best_iou = 0.0
            best_y_idx = -1

            # Find YOLO detection Y having the same class and highest IoU
            for y_idx, y_det in enumerate(yolo_cands):
                if y_idx in matched_yolo_indices:
                    continue
                if y_det["class_name"] != r_class:
                    continue

                iou = calculate_iou(r_box, y_det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_y_idx = y_idx

            # If IoU >= fusion_threshold: merge
            if best_iou >= self.iou_threshold and best_y_idx >= 0:
                y_det = yolo_cands[best_y_idx]
                y_box = y_det["box"]
                y_score = y_det["score"]

                # Confidence-weighted box averaging:
                # x = (c1*x1 + c2*x2) / (c1 + c2)
                weight_sum = r_score + y_score
                fused_box = [
                    (r_score * r_box[0] + y_score * y_box[0]) / weight_sum,
                    (r_score * r_box[1] + y_score * y_box[1]) / weight_sum,
                    (r_score * r_box[2] + y_score * y_box[2]) / weight_sum,
                    (r_score * r_box[3] + y_score * y_box[3]) / weight_sum,
                ]

                # Calibrated combined confidence score
                # Multi-detector consensus bonus:
                agreement_bonus = 1.0 + (self.bonus_factor - 1.0) * best_iou
                fused_score = min(1.0, ((r_score + y_score) / 2.0) * agreement_bonus)

                if fused_score >= self.fused_conf_thresh:
                    fused_detections.append({
                        "class_name": r_class,
                        "box": fused_box,
                        "score": round(float(fused_score), 4),
                        "source": "Fused (R-CNN + YOLOv3)",
                        "iou_overlap": round(float(best_iou), 4)
                    })

                matched_yolo_indices.add(best_y_idx)
            else:
                # Unmatched R-CNN detection retained
                if r_score >= self.fused_conf_thresh:
                    fused_detections.append({
                        "class_name": r_class,
                        "box": list(r_box),
                        "score": round(float(r_score), 4),
                        "source": "Faster R-CNN Sole",
                        "iou_overlap": 0.0
                    })

        # Step 3: Add all unmatched YOLO detections
        for y_idx, y_det in enumerate(yolo_cands):
            if y_idx not in matched_yolo_indices:
                if y_det["score"] >= self.fused_conf_thresh:
                    fused_detections.append({
                        "class_name": y_det["class_name"],
                        "box": list(y_det["box"]),
                        "score": round(float(y_det["score"]), 4),
                        "source": "YOLOv3 Sole",
                        "iou_overlap": 0.0
                    })

        # Sort final fused detections by confidence descending
        fused_detections.sort(key=lambda x: x["score"], reverse=True)
        return fused_detections

    def fuse_nms_baseline(self, faster_rcnn_dets, yolov3_dets, nms_iou_thresh=0.5):
        """
        Baseline 1: Standard Non-Maximum Suppression (NMS) across concatenated detections.
        Keeps the box with higher confidence and discards the other.
        """
        all_dets = faster_rcnn_dets + yolov3_dets
        all_dets = [d for d in all_dets if d["score"] >= self.fused_conf_thresh]
        all_dets.sort(key=lambda x: x["score"], reverse=True)

        keep = []
        while len(all_dets) > 0:
            top = all_dets.pop(0)
            keep.append({
                "class_name": top["class_name"],
                "box": top["box"],
                "score": top["score"],
                "source": f"NMS ({top.get('source', 'Model')})"
            })

            all_dets = [
                d for d in all_dets
                if not (d["class_name"] == top["class_name"] and calculate_iou(d["box"], top["box"]) >= nms_iou_thresh)
            ]

        return keep

    def fuse_wbf_baseline(self, faster_rcnn_dets, yolov3_dets, iou_thresh=0.5):
        """
        Baseline 2: Weighted Boxes Fusion (WBF) style clustering.
        Clusters overlapping boxes of the same class and averages coordinates.
        """
        all_dets = faster_rcnn_dets + yolov3_dets
        all_dets = [d for d in all_dets if d["score"] >= self.fused_conf_thresh]

        by_class = {}
        for d in all_dets:
            by_class.setdefault(d["class_name"], []).append(d)

        output = []
        for cname, dets in by_class.items():
            dets.sort(key=lambda x: x["score"], reverse=True)
            clusters = []

            for d in dets:
                matched_cluster = None
                for c in clusters:
                    # Check IoU against cluster representative
                    if calculate_iou(d["box"], c["box"]) >= iou_thresh:
                        matched_cluster = c
                        break

                if matched_cluster:
                    matched_cluster["boxes"].append(d["box"])
                    matched_cluster["scores"].append(d["score"])
                    # Recompute weighted average
                    ws = sum(matched_cluster["scores"])
                    new_box = [
                        sum(b[i] * s for b, s in zip(matched_cluster["boxes"], matched_cluster["scores"])) / ws
                        for i in range(4)
                    ]
                    matched_cluster["box"] = new_box
                else:
                    clusters.append({
                        "box": list(d["box"]),
                        "boxes": [d["box"]],
                        "scores": [d["score"]],
                        "class_name": cname
                    })

            for c in clusters:
                avg_score = sum(c["scores"]) / len(c["scores"])
                output.append({
                    "class_name": cname,
                    "box": c["box"],
                    "score": round(float(avg_score), 4),
                    "source": "WBF Baseline"
                })

        output.sort(key=lambda x: x["score"], reverse=True)
        return output
