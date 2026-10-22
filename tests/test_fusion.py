import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Unit tests for Detection Fusion Algorithm:
- Test 1: Overlapping boxes of same class are merged with weighted average coordinates.
- Test 2: Overlapping boxes of different classes are NOT merged.
- Test 3: Disjoint boxes (IoU < threshold) are both preserved as sole detections.
- Test 4: Weighted averaging pulls the fused box towards the higher-confidence detector.
- Test 5: Empty inputs / single model detections handled gracefully.
"""

# import pytest
import config
from fusion.detection_fusion import DetectionFusion
from utils.metrics import calculate_iou

def test_same_class_box_merging():
    fusion = DetectionFusion(iou_threshold=0.5, bonus_factor=1.1)
    
    # Ground truth object at [100, 100, 200, 200]
    rcnn_dets = [{"class_name": "Bottle", "box": [98.0, 102.0, 198.0, 202.0], "score": 0.90}]
    yolo_dets = [{"class_name": "Bottle", "box": [102.0, 98.0, 202.0, 198.0], "score": 0.80}]

    fused = fusion.fuse(rcnn_dets, yolo_dets)

    assert len(fused) == 1, "Should merge 2 overlapping same-class boxes into exactly 1 fused detection."
    f = fused[0]
    assert f["class_name"] == "Bottle"
    assert "Fused" in f["source"]
    assert f["score"] > 0.85, "Fused score should reflect consensus agreement bonus."
    # Bounding box should be weighted mean
    expected_x1 = (0.90 * 98.0 + 0.80 * 102.0) / (0.90 + 0.80)
    assert abs(f["box"][0] - expected_x1) < 1e-4

def test_different_class_no_merging():
    fusion = DetectionFusion(iou_threshold=0.5)
    
    rcnn_dets = [{"class_name": "Bottle", "box": [100.0, 100.0, 200.0, 200.0], "score": 0.85}]
    yolo_dets = [{"class_name": "Can", "box": [100.0, 100.0, 200.0, 200.0], "score": 0.75}]

    fused = fusion.fuse(rcnn_dets, yolo_dets)

    assert len(fused) == 2, "Different classes must not be merged even with high IoU."
    class_names = {d["class_name"] for d in fused}
    assert class_names == {"Bottle", "Can"}

def test_disjoint_boxes_retained():
    fusion = DetectionFusion(iou_threshold=0.5)
    
    rcnn_dets = [{"class_name": "Can", "box": [10.0, 10.0, 50.0, 50.0], "score": 0.80}]
    yolo_dets = [{"class_name": "Can", "box": [300.0, 300.0, 400.0, 400.0], "score": 0.70}]

    fused = fusion.fuse(rcnn_dets, yolo_dets)

    assert len(fused) == 2, "Disjoint boxes of the same class must both be preserved."
    assert any(d["source"] == "Faster R-CNN Sole" for d in fused)
    assert any(d["source"] == "YOLOv3 Sole" for d in fused)

def test_asymmetric_confidence_weighting():
    fusion = DetectionFusion(iou_threshold=0.4)
    
    # Model 1 has high confidence, Model 2 has low confidence
    rcnn_dets = [{"class_name": "Carton", "box": [100.0, 100.0, 200.0, 200.0], "score": 0.90}]
    yolo_dets = [{"class_name": "Carton", "box": [120.0, 120.0, 220.0, 220.0], "score": 0.40}]

    fused = fusion.fuse(rcnn_dets, yolo_dets)
    assert len(fused) == 1
    f_box = fused[0]["box"]
    
    # Distance to R-CNN box should be smaller than distance to YOLO box
    dist_rcnn = abs(f_box[0] - 100.0)
    dist_yolo = abs(f_box[0] - 120.0)
    assert dist_rcnn < dist_yolo, "Fused box must be biased towards higher confidence detector."

def test_empty_inputs():
    fusion = DetectionFusion()
    assert fusion.fuse([], []) == []
    
    rcnn_dets = [{"class_name": "Cup", "box": [50.0, 50.0, 100.0, 100.0], "score": 0.85}]
    res = fusion.fuse(rcnn_dets, [])
    assert len(res) == 1
    assert res[0]["class_name"] == "Cup"

if __name__ == "__main__":
    test_same_class_box_merging()
    test_different_class_no_merging()
    test_disjoint_boxes_retained()
    test_asymmetric_confidence_weighting()
    test_empty_inputs()
    print("ALL FUSION UNIT TESTS PASSED!")
