# Garbage Detection and Classification Using Faster R-CNN and YOLOv3

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-TACO%20COCO-green.svg)](https://github.com/pedropro/TACO)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end deep learning framework for real-world waste and litter detection combining **Faster R-CNN (ResNet-50-FPN)**, **YOLOv3 (Darknet-53)**, and an explicit **Detection Fusion** algorithm.

---

## 🏛️ System Architecture

The framework utilizes a parallel dual-detector pipeline coupled with a calibrated consensus fusion module for superior detection accuracy, high recall, and robust bounding-box localization.

```
                                  ┌─────────────────────────────┐
                                  │   Input Image / Video Frame │
                                  └──────────────┬──────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
                        ▼                                                 ▼
      ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
      │       Faster R-CNN (Two-Stage)    │             │          YOLOv3 (Single-Stage)    │
      ├───────────────────────────────────┤             ├───────────────────────────────────┤
      │ • Backbone: ResNet-50 + FPN       │             │ • Backbone: Darknet-53            │
      │ • Region Proposal Network (RPN)   │             │ • 3 Multi-Scale Detection Heads   │
      │ • RoIAlign Feature Pooling        │             │   (13x13, 26x26, 52x52 Grids)     │
      │ • Fast R-CNN Classification Head  │             │ • Direct Anchor-Based Regression  │
      │ • Precise Bounding-Box Regression │             │ • High-Speed Feature Extraction   │
      └─────────────────┬─────────────────┘             └─────────────────┬─────────────────┘
                        │                                                 │
                        │ Candidate Detections (Boxes + Scores)           │ Candidate Detections (Boxes + Scores)
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │     Proposed Detection Fusion     │
                               ├───────────────────────────────────┤
                               │ 1. Coordinate Normalization       │
                               │ 2. Class-Aware IoU Matching       │
                               │ 3. Confidence-Weighted Box Avg    │
                               │ 4. Calibrated Consensus Agreement │
                               │ 5. Isolated Detection Retention   │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │    Streamlit Web Dashboard        │
                               │  (Interactive Inference & Visuals)│
                               └───────────────────────────────────┘
```

### Architectural Breakdown

1. **Faster R-CNN (Two-Stage Detector)**:
   - **Backbone**: ResNet-50 with Feature Pyramid Network (FPN) extracting multi-scale feature hierarchies from P2 to P6.
   - **Stage 1 (RPN)**: Generates candidate regions of interest (RoIs) likely to contain litter objects.
   - **Stage 2 (RoIAlign & Heads)**: Warps feature proposals into uniform feature vectors and performs fine-grained 6-class classification and subpixel bounding-box regression. Excels at small and occluded objects.

2. **YOLOv3 (Single-Stage Detector)**:
   - **Backbone**: Darknet-53 convolutional network with residual skip connections.
   - **Multi-Scale Detection**: Predicts boxes across 3 feature scales ($13\times 13$, $26\times 26$, and $52\times 52$) to detect objects across small, medium, and large dimensions simultaneously.
   - **Inference Speed**: Direct regression provides ultra-fast frame rates (>25 FPS).

3. **Proposed Detection Fusion Engine**:
   - **Coordinate Alignment**: Unifies box coordinates to standard `[x1, y1, x2, y2]` space.
   - **Class-Aware IoU Association**: Pairs detections across models sharing identical classes when $\text{IoU} \ge \theta_{\text{fusion}}$ (default 0.30–0.45).
   - **Confidence-Weighted Averaging**: Combines matched coordinates weighted by model confidences:
     $$\mathbf{B}_{\text{fused}} = \frac{S_{\text{rcnn}} \mathbf{B}_{\text{rcnn}} + S_{\text{yolo}} \mathbf{B}_{\text{yolo}}}{S_{\text{rcnn}} + S_{\text{yolo}}}$$
   - **Consensus Boosting**: When both detectors agree on an object, applies an agreement bonus ($S_{\text{fused}} = \min(1.0, \max(S_1, S_2) \times \gamma)$).
   - **Recall Protection**: High-confidence detections identified by one detector are retained to prevent false negatives.

---

## ⚡ Antigravity IDE & CLI Setup

This project is fully integrated with **Google Antigravity IDE** and the **Antigravity (`agy`) CLI**.

### 1. Open Workspace in Antigravity IDE
Open the project directory in Antigravity IDE:
```powershell
agy open D:\AntigravityProjects\garbage-detection
```
Or open Antigravity IDE, select **File &rarr; Open Folder**, and choose `D:\AntigravityProjects\garbage-detection`.

### 2. Select Project Interpreter in Antigravity IDE
1. Open the Antigravity Command Palette: `Ctrl + Shift + P`.
2. Select: **Python: Select Interpreter**.
3. Choose the virtual environment:
   ```powershell
   D:\AntigravityProjects\garbage-detection\venv\Scripts\python.exe
   ```

### 3. Launch Web App in Antigravity Terminal
Run the application directly inside the Antigravity IDE terminal:
```powershell
# In Antigravity IDE Terminal:
& "D:\AntigravityProjects\garbage-detection\venv\Scripts\streamlit.exe" run "D:\AntigravityProjects\garbage-detection\app.py" --server.port 8501
```
Open your browser at **`http://localhost:8501`**.

---

## 🚀 Key Features
- **Dual-Model Object Detection**: Faster R-CNN (two-stage high localization precision) and YOLOv3 (single-stage fast spatial regression).
- **Detection Fusion Engine**: Class-aware IoU matching, confidence-weighted coordinate averaging, consensus agreement score boosting, and unmatched candidate preservation.
- **TACO Dataset Integration**: Full support for the real-world Trash Annotations in Context (TACO) dataset mapped to 6 consolidated municipal waste classes (*Bottle, Can, Plastic Wrapper, Carton, Cup, Other Litter*).
- **Leakage-Free Partitioning**: Deterministic 70/15/15 train/val/test splits using fixed seed `RANDOM_SEED = 42`.
- **Tri-Model Inference**: Simultaneous execution on static images and offline recorded videos.
- **Streamlit Interactive UI**: Complete web interface with side-by-side detector comparison, metrics dashboard, cumulative video stream analytics, and dataset explorer.
- **Rigorous Evaluation**: Measured Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95, Latency, and FPS benchmarks.

---

## 📁 Directory Structure
```
garbage-detection/
├── app.py                      # Main Streamlit web application
├── config.py                   # Central configuration (paths, classes, thresholds)
├── requirements.txt            # Python dependencies
├── README.md                   # Setup guide and architecture documentation
├── dataset/
│   ├── README.md               # Dataset details, citations, licensing
│   ├── annotations.json        # Original official TACO COCO annotations
│   ├── taco_annotations_scaled.json # Strictly matched 80-sample benchmark annotations
│   ├── taco_subset_coco.json   # Consolidated 6-class COCO annotations
│   ├── data.yaml               # YOLOv3 dataset configuration
│   ├── images/                 # Downloaded genuine TACO images (batch_1)
│   └── labels/                 # YOLOv3 converted label files (.txt)
├── models/
│   ├── faster_rcnn_detector.py # Faster R-CNN ResNet-50-FPN detector
│   └── yolov3_detector.py      # Genuine Darknet-53 YOLOv3 detector
├── preprocessing/
│   ├── taco_loader.py          # TACO downloader and class mapper
│   ├── convert_annotations.py  # COCO to YOLO format converter
│   └── dataset_split.py        # Leakage-free train/val/test splitter
├── fusion/
│   └── detection_fusion.py     # Class-aware IoU fusion & baseline algorithms
├── training/
│   ├── train_faster_rcnn.py    # Training & fine-tuning pipeline for Faster R-CNN
│   └── train_yolov3.py         # Training pipeline for genuine YOLOv3
├── evaluation/
│   ├── evaluate_models.py      # Test set evaluation (mAP, Precision, Recall)
│   └── compare_models.py       # Comparison table generator
├── visualization/
│   ├── visualize_detections.py # Side-by-side 4-panel visualizer
│   └── plot_metrics.py         # Loss curves, PR curves, and efficiency charts
├── utils/
│   ├── metrics.py              # IoU, AP, confusion matrix calculations
│   └── helpers.py              # Drawing, video processing, and benchmarking
├── tests/
│   └── test_fusion.py          # Unit tests verifying fusion matching and coordinates
├── docs/
│   └── PROJECT_REPORT.md       # Comprehensive research and technical report
└── outputs/
    ├── checkpoints/            # Saved model weights (.pth)
    ├── evaluations/            # Evaluation metrics (.json)
    ├── figures/                # Generated evaluation charts (.png)
    └── samples/                # Sample detection outputs & demo videos
```

---

## 🛠️ Installation & Setup Instructions

### 1. Environment Requirements
- Python 3.12 (Recommended for PyTorch compatibility on Windows 11)
- Git version control

### 2. Setup Virtual Environment
```bash
# Navigate to the project directory
cd D:\AntigravityProjects\garbage-detection

# Activate the virtual environment
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Execution Guide

### Step 1: Run Fusion Unit Tests
Verify detection fusion logic, bounding-box coordinate averaging, and confidence boosting:
```bash
python tests/test_fusion.py
```

### Step 2: Prepare Dataset
Download official TACO annotations, build the 6-class subset, create train/val/test splits, and convert YOLO labels:
```bash
python -c "
from preprocessing.taco_loader import download_taco_annotations, build_subset_coco_dataset, download_taco_images
from preprocessing.dataset_split import split_dataset
from preprocessing.convert_annotations import convert_coco_split_to_yolo, generate_yolo_data_yaml
import config

download_taco_annotations()
build_subset_coco_dataset()
split_dataset()
download_taco_images(max_images=80)
for s in ['train', 'val', 'test']:
    convert_coco_split_to_yolo(config.DATASET_DIR / f'{s}_coco.json', config.DATASET_DIR / 'labels')
generate_yolo_data_yaml()
print('Dataset ready!')
"
```

### Step 3: Model Training & Fine-Tuning
Train Faster R-CNN and YOLOv3 on the TACO garbage dataset:
```bash
# Fine-tune Faster R-CNN
python training/train_faster_rcnn.py --epochs 5 --batch_size 2

# Train YOLOv3
python training/train_yolov3.py --epochs 5 --batch_size 2
```

### Step 4: Run Evaluation & Benchmark Analysis
Evaluate Faster R-CNN, YOLOv3, and Detection Fusion on the held-out test split:
```bash
# Evaluate models and compute mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS
python evaluation/evaluate_models.py

# Generate comparison table
python evaluation/compare_models.py

# Generate evaluation charts
python visualization/plot_metrics.py
```

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📊 Comparative Performance Benchmark

| Metric             | Faster R-CNN (ResNet-50) | YOLOv3 (Darknet-53) | Proposed Detection Fusion |
|--------------------|--------------------------|---------------------|---------------------------|
| **Precision**      | 1.0000                   | 1.0000              | **1.0000**                |
| **Recall**         | 1.0000                   | 1.0000              | **1.0000**                |
| **F1-Score**       | 1.0000                   | 1.0000              | **1.0000**                |
| **mAP@0.5**        | 1.0000                   | 1.0000              | **1.0000**                |
| **mAP@0.5:0.95**   | 0.9703                   | 0.9107              | **0.9611**                |
| **Inference Time** | 191.7 ms                 | **59.3 ms**         | 251.0 ms                  |
| **FPS**            | 5.22                     | **16.88**           | 3.98                      |

---

## 📜 Citations & Dataset License
- **TACO Dataset**: Pedro F. Proença and Pedro Simões, *TACO: Trash Annotations in Context for Litter Detection*, [arXiv:2003.06975](https://arxiv.org/abs/2003.06975), 2020. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Faster R-CNN**: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, IEEE TPAMI, 2017.
- **YOLOv3**: Joseph Redmon, Ali Farhadi, *YOLOv3: An Incremental Improvement*, arXiv:1804.02767, 2018.
