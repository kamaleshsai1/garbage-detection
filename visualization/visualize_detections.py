"""
Detection Visualization Module:
- Renders individual detector predictions (Faster R-CNN, YOLOv3)
- Renders Detection Fusion output
- Produces side-by-side 4-panel comparison images:
  [Original Input | Faster R-CNN | YOLOv3 | Fused Result]
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.helpers import draw_detections, load_image

def create_side_by_side_comparison(
    image_input,
    rcnn_detections,
    yolo_detections,
    fused_detections,
    output_path=None
):
    """
    Creates a 2x2 or 1x4 side-by-side comparison image displaying:
    1. Original Input Image
    2. Faster R-CNN Detections
    3. YOLOv3 Detections
    4. Detection Fusion Output
    """
    rgb_img, _ = load_image(image_input)
    h, w = rgb_img.shape[:2]

    # Render each panel
    p1 = rgb_img.copy()
    cv2.putText(p1, "1. Original Input", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    p2 = draw_detections(rgb_img, rcnn_detections, title="2. Faster R-CNN")
    p3 = draw_detections(rgb_img, yolo_detections, title="3. YOLOv3")
    p4 = draw_detections(rgb_img, fused_detections, title="4. Proposed Fusion")

    # Stack as 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB to BGR for cv2 write
        cv2.imwrite(str(out_p), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"[Visualization] Side-by-side comparison saved to: {out_p}")

    return grid
