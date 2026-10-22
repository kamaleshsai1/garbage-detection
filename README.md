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

```mermaid
flowchart TD
    subgraph INPUT["1. Input Data Stream"]
        IMG["Input Image / Video Frame<br/>(RGB 640x480)"]
    end

    subgraph FRCNN["2. Faster R-CNN (Two-Stage Detector)"]
        RN50["ResNet-50 Backbone<br/>(Deep Residual Feature Extraction)"]
        FPN["Feature Pyramid Network (FPN)<br/>[P2, P3, P4, P5, P6 Hierarchy]"]
        RPN["Region Proposal Network (RPN)<br/>(Candidate Anchor Proposals)"]
        ROI["RoIAlign Layer<br/>(Spatial Feature Warping)"]
        FHEAD["Fast R-CNN Detection Head<br/>• 6-Class Softmax Classification<br/>• Bounding-Box Coordinate Regression"]
        
        RN50 --> FPN
        FPN --> RPN
        RPN -->|Candidate RoIs| ROI
        FPN --> ROI
        ROI --> FHEAD
    end

    subgraph YOLO["3. YOLOv3 (Single-Stage Detector)"]
        DN53["Darknet-53 Backbone<br/>(53-Layer Residual Convolutional Network)"]
        MSF["Multi-Scale Feature Extractor<br/>(FPN-like Skip Concat)"]
        S1["Scale 1: 52x52 Grid<br/>(Small Litter Detection)"]
        S2["Scale 2: 26x26 Grid<br/>(Medium Litter Detection)"]
        S3["Scale 3: 13x13 Grid<br/>(Large Litter Detection)"]
        
        DN53 --> MSF
        MSF --> S1
        MSF --> S2
        MSF --> S3
    end

    subgraph FUSION["4. Proposed Detection Fusion Engine"]
        MATCH["Class-Aware IoU Association<br/>(Threshold &theta; = 0.30 - 0.45)"]
        WAVG["Confidence-Weighted Coordinate Averaging<br/>B_fused = (S1·B1 + S2·B2) / (S1 + S2)"]
        BONUS["Consensus Score Calibration<br/>S_fused = min(1.0, max(S1, S2) &times; &gamma;)"]
        ISO["Single-Detector Retention<br/>(Recall Preservation Safeguard)"]
        
        MATCH --> WAVG
        WAVG --> BONUS
        MATCH --> ISO
    end

    subgraph OUTPUT["5. Web Application Dashboard (Streamlit)"]
        DASH["Streamlit Interactive UI<br/>(http://localhost:8501)"]
        PANEL1["Side-by-Side Model Comparison<br/>(Faster R-CNN vs. YOLOv3)"]
        PANEL2["Proposed Fused Output Display<br/>(Color Badges & Box Inventory)"]
        PANEL3["Video Analytics & Benchmark Metrics<br/>(Frame Stream & Evaluation Charts)"]
        
        DASH --> PANEL1
        DASH --> PANEL2
        DASH --> PANEL3
    end

    IMG --> RN50
    IMG --> DN53
    FHEAD -->|Detections R1| MATCH
    S1 & S2 & S3 -->|Detections R2| MATCH
    BONUS --> DASH
    ISO --> DASH

    style INPUT fill:#161b22,stroke:#30363d,stroke-width:2px,color:#c9d1d9
    style FRCNN fill:#0d1117,stroke:#1f6feb,stroke-width:2px,color:#c9d1d9
    style YOLO fill:#0d1117,stroke:#d29922,stroke-width:2px,color:#c9d1d9
    style FUSION fill:#0d1117,stroke:#2ea043,stroke-width:2px,color:#c9d1d9
    style OUTPUT fill:#161b22,stroke:#8957e5,stroke-width:2px,color:#c9d1d9
```

### Architectural Breakdown

```
Input Image / Video Frame
          │
          ├───► Faster R-CNN ──► ResNet-50 ──► FPN ──► RPN ──► RoIAlign ──► Head ──┐
          │                                                                         ├──► Detection Fusion ──► Web UI
          └───► YOLOv3       ──► Darknet-53 ──► Multi-Scale Heads (52, 26, 13) ────┘
```

1. **Faster R-CNN (Two-Stage Detector)**:
   - **Backbone**: ResNet-50 with Feature Pyramid Network (FPN) extracting multi-scale feature representations across pyramids P2 through P6.
   - **Stage 1 (Region Proposal Network)**: Slides a small network over convolutional feature maps to generate candidate object bounding boxes independent of classes.
   - **Stage 2 (RoIAlign & Prediction Heads)**: Preserves spatial fidelity via RoIAlign and outputs class probabilities across the 6 waste categories along with refined subpixel bounding-box offsets. Excels in dense, occluded, and complex backgrounds.

2. **YOLOv3 (Single-Stage Detector)**:
   - **Backbone**: Darknet-53 feature extractor incorporating 53 convolutional layers and residual shortcut connections.
   - **Multi-Scale Heads**: Predicts boxes across 3 distinct feature grids:
     - **$52\times 52$**: High-resolution map tailored for small litter items (caps, bottle tabs).
     - **$26\times 26$**: Medium-scale map for standard packaging (cups, cans, snack wrappers).
     - **$13\times 13$**: Coarse map for large objects (cardboard cartons, large bottles).
   - **Inference Speed**: Direct single-stage regression delivers fast frame rates (>25 FPS).

3. **Proposed Detection Fusion Engine**:
   - **Coordinate Normalization**: Normalizes all predictions to a unified $[x_1, y_1, x_2, y_2]$ coordinate space.
   - **Class-Aware IoU Matching**: Associates pairs of detections $(d_i^{\text{rcnn}}, d_j^{\text{yolo}})$ if and only if they share identical predicted class labels and satisfy $\text{IoU}(d_i, d_j) \ge \theta_{\text{fusion}}$ (default 0.30–0.45).
   - **Confidence-Weighted Bounding Box Averaging**: Calculates consensus bounding box coordinates weighted by individual model detection confidences:
     $$\mathbf{B}_{\text{fused}} = \frac{S_{\text{rcnn}} \mathbf{B}_{\text{rcnn}} + S_{\text{yolo}} \mathbf{B}_{\text{yolo}}}{S_{\text{rcnn}} + S_{\text{yolo}}}$$
   - **Consensus Score Calibration**: When both detectors validate an object candidate, a consensus agreement bonus factor $\gamma \in [1.10, 1.25]$ boosts the confidence score:
     $$S_{\text{fused}} = \min\left(1.0, \max(S_{\text{rcnn}}, S_{\text{yolo}}) \times \gamma\right)$$
   - **Recall Safeguard (Isolated Retention)**: Detections above confidence threshold identified by only one model are retained, ensuring high recall on difficult objects.

---

## 💻 VS Code Setup Guide

This project is configured for setup and debugging in **Visual Studio Code**.

### 1. Open the Project in VS Code
Launch VS Code and open the repository folder:
```bash
code D:\AntigravityProjects\garbage-detection
```
Or use the menu: **File** &rarr; **Open Folder...** &rarr; select `garbage-detection`.

### 2. Configure Python Interpreter in VS Code
1. Open the Command Palette: `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS).
2. Type: **Python: Select Interpreter**.
3. Select the project virtual environment:
   ```
   .\venv\Scripts\python.exe
   ```
   *(VS Code will automatically detect the `.venv` or `venv` directory).*

### 3. Open Integrated Terminal in VS Code
Open a new integrated terminal (`Ctrl + \`` or **Terminal** &rarr; **New Terminal**) and activate the environment:
```powershell
# Windows PowerShell in VS Code:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat

# Linux / macOS:
source venv/bin/activate
```

---

## 🚀 Key Features
- **Dual-Model Object Detection**: Faster R-CNN (two-stage high localization precision) and YOLOv3 (single-stage fast spatial regression).
- **Detection Fusion Engine**: Class-aware IoU matching, confidence-weighted coordinate averaging, consensus agreement score boosting, and unmatched candidate preservation.
- **TACO Dataset Integration**: Full support for the real-world Trash Annotations in Context (TACO) dataset mapped to 6 consolidated municipal waste classes (*Bottle, Can, Plastic Wrapper, Carton, Cup, Other Litter*).
- **Leakage-Free Partitioning**: Deterministic 70/15/15 train/val/test splits using fixed seed `RANDOM_SEED = 42`.
- **Dual Input Modes**: Simultaneous execution on static images and offline recorded videos.
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
│   ├── visualize_detections.py # Side-by-side visualizer
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
    ├── evaluations/            # Evaluation metrics (.json, .md)
    ├── figures/                # Generated evaluation charts (.png)
    └── samples/                # Sample detection outputs & demo videos
```

---

## 🛠️ Installation & Setup Instructions

### 1. Prerequisites
- Python 3.12 (Recommended for PyTorch compatibility on Windows 11)
- Git version control

### 2. Setup Virtual Environment
```bash
# Navigate to the project directory
cd D:\AntigravityProjects\garbage-detection

# Create a virtual environment (if not already created)
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows CMD:
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Execution Guide (in VS Code Terminal)

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
Run the Streamlit server from the VS Code terminal:
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
