"""
Helper utilities for image loading, preprocessing, drawing bounding boxes,
video capture processing, and latency benchmarking.
"""

import time
import cv2
import numpy as np
from PIL import Image
import config

def load_image(image_input):
    """
    Loads an image from filepath, PIL Image, or bytes and returns:
    - rgb_img: numpy array uint8 (H, W, 3) in RGB
    - bgr_img: numpy array uint8 (H, W, 3) in BGR
    """
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, bgr
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return rgb, bgr
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            bgr = cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
        else:
            rgb = image_input
            bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        return rgb, bgr
    else:
        # Byte stream / file buffer
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Unable to decode uploaded image file.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, bgr

def draw_detections(image, detections, title=None, score_thresh=0.0):
    """
    Draws modern bounding boxes, colored tags, and confidence labels on image.
    image: RGB numpy array
    detections: list of dicts: {'box': [x1, y1, x2, y2], 'class_name': str, 'score': float}
    Returns annotated RGB image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Optional header title
    if title:
        cv2.putText(annotated, title, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

    for det in detections:
        score = det.get("score", 1.0)
        if score < score_thresh:
            continue

        box = det["box"]
        cname = det.get("class_name", "Object")
        
        x1 = int(max(0, min(w - 1, box[0])))
        y1 = int(max(0, min(h - 1, box[1])))
        x2 = int(max(0, min(w - 1, box[2])))
        y2 = int(max(0, min(h - 1, box[3])))

        if x2 <= x1 or y2 <= y1:
            continue

        color = config.CLASS_COLORS.get(cname, (0, 255, 0))
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw label tag
        label = f"{cname} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        tag_y1 = max(0, y1 - label_h - 6)
        tag_y2 = y1
        tag_x2 = min(w, x1 + label_w + 8)

        # Solid tag background
        cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        # White text on tag
        cv2.putText(annotated, label, (x1 + 4, tag_y2 - 4), font, 
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated

class BenchmarkTimer:
    """Measures execution latency and calculates frames per second (FPS)."""
    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.elapsed_ms = (end_time - self.start_time) * 1000.0

    @property
    def fps(self):
        return (1000.0 / self.elapsed_ms) if self.elapsed_ms > 0 else 0.0
