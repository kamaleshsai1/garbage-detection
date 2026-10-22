"""
Evaluation metrics computation module:
- IoU calculation (single and vectorized)
- Precision, Recall, F1-score
- Average Precision (AP) and 11-point / continuous interpolation
- mAP@0.5 and mAP@0.5:0.95 (COCO standard)
- Per-class evaluation and Confusion Matrix / Detection Error Analysis
"""

import numpy as np

def calculate_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using the continuous area under the PR curve (VOC/COCO standard).
    recalls: 1D array of recall values sorted ascending
    precisions: 1D array of precision values
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)

def evaluate_class_detections(all_gt_boxes, all_pred_boxes, iou_thresh=0.5):
    """
    Evaluates detections for a single class across an entire test set.
    all_gt_boxes: list of lists of GT boxes per image: [[box1, box2, ...], ...]
    all_pred_boxes: list of detections across all images:
        [{'image_idx': int, 'box': [x1, y1, x2, y2], 'score': float}, ...]
    """
    total_gts = sum(len(boxes) for boxes in all_gt_boxes)
    if total_gts == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "tp": 0, "fp": 0, "fn": 0}

    if len(all_pred_boxes) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "tp": 0, "fp": 0, "fn": total_gts}

    # Sort all predictions by confidence score descending
    sorted_preds = sorted(all_pred_boxes, key=lambda x: x["score"], reverse=True)

    # Track matched GTs per image
    gt_matched = {img_idx: [False] * len(all_gt_boxes[img_idx]) for img_idx in range(len(all_gt_boxes))}

    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    for p_idx, pred in enumerate(sorted_preds):
        img_idx = pred["image_idx"]
        p_box = pred["box"]
        gts = all_gt_boxes[img_idx]

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gts):
            iou = calculate_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh:
            if not gt_matched[img_idx][best_gt_idx]:
                tp[p_idx] = 1
                gt_matched[img_idx][best_gt_idx] = True
            else:
                fp[p_idx] = 1  # Duplicate detection of same ground-truth
        else:
            fp[p_idx] = 1

    # Cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / float(total_gts)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

    final_tp = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
    final_fp = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
    final_fn = total_gts - final_tp

    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
    final_f1 = (2 * final_precision * final_recall / (final_precision + final_recall)
                if (final_precision + final_recall) > 0 else 0.0)

    ap = compute_ap(recalls, precisions)

    return {
        "precision": final_precision,
        "recall": final_recall,
        "f1": final_f1,
        "ap": ap,
        "tp": final_tp,
        "fp": final_fp,
        "fn": final_fn,
        "recalls": recalls.tolist(),
        "precisions": precisions.tolist()
    }

def evaluate_all_classes(dataset_gts, model_predictions, class_names, iou_thresh=0.5):
    """
    Computes overall and per-class precision, recall, F1, and mAP@iou_thresh.
    dataset_gts: list of images, each image has list of {'class_name': str, 'box': [x1, y1, x2, y2]}
    model_predictions: list of images, each image has list of {'class_name': str, 'box': [x1, y1, x2, y2], 'score': float}
    """
    per_class_results = {}
    aps = []

    for cname in class_names:
        # Extract GTs for this class
        all_gt_boxes = []
        for img in dataset_gts:
            c_gts = [item["box"] for item in img if item["class_name"] == cname]
            all_gt_boxes.append(c_gts)

        # Extract predictions for this class
        all_pred_boxes = []
        for img_idx, img_preds in enumerate(model_predictions):
            for pred in img_preds:
                if pred["class_name"] == cname:
                    all_pred_boxes.append({
                        "image_idx": img_idx,
                        "box": pred["box"],
                        "score": pred["score"]
                    })

        res = evaluate_class_detections(all_gt_boxes, all_pred_boxes, iou_thresh=iou_thresh)
        per_class_results[cname] = res
        aps.append(res["ap"])

    mean_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0
    overall_tp = sum(res["tp"] for res in per_class_results.values())
    overall_fp = sum(res["fp"] for res in per_class_results.values())
    overall_fn = sum(res["fn"] for res in per_class_results.values())

    overall_prec = (overall_tp / (overall_tp + overall_fp)) if (overall_tp + overall_fp) > 0 else 0.0
    overall_rec = (overall_tp / (overall_tp + overall_fn)) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "mAP": mean_ap,
        "precision": float(overall_prec),
        "recall": float(overall_rec),
        "f1": float(overall_f1),
        "total_tp": overall_tp,
        "total_fp": overall_fp,
        "total_fn": overall_fn,
        "per_class": per_class_results
    }

def evaluate_map_coco_range(dataset_gts, model_predictions, class_names):
    """
    Computes mAP@0.5:0.95 across 10 IoU thresholds (0.50, 0.55, ..., 0.95).
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    map_scores = []
    
    for thresh in iou_thresholds:
        res = evaluate_all_classes(dataset_gts, model_predictions, class_names, iou_thresh=thresh)
        map_scores.append(res["mAP"])

    map_50 = map_scores[0]
    map_50_95 = float(np.mean(map_scores))
    return map_50, map_50_95
