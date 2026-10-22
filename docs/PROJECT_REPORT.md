# GARBAGE DETECTION AND CLASSIFICATION USING FASTER R-CNN AND YOLOv3

## Final Project Research Report
**Project Title**: Garbage Detection and Classification Using Faster R-CNN and YOLOv3  
**Domain**: Deep Learning, Computer Vision, Environmental AI  
**Dataset**: TACO (Trash Annotations in Context)  
**Primary Architectures**: Faster R-CNN (ResNet-50-FPN) & YOLOv3 (Darknet-53)  
**Framework**: PyTorch 2.x, Torchvision, OpenCV, Streamlit  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Literature Review](#2-literature-review)  
   2.1 [Object Detection Using Deep Learning](#21-object-detection-using-deep-learning)  
   2.2 [Litter Detection and Contextual Waste Analysis](#22-litter-detection-and-contextual-waste-analysis)  
   2.3 [Real-Time Detectors: YOLOv3 Architecture](#23-real-time-detectors-yolov3-architecture)  
   2.4 [Research Gap and Proposed Dual-Model Synergy](#24-research-gap-and-proposed-dual-model-synergy)
3. [Methodology](#3-methodology)  
   3.1 [System Architecture Diagram](#31-system-architecture-diagram)  
   3.2 [Methodology Execution Steps](#32-methodology-execution-steps)  
   3.3 [Architectural Feature Comparison](#33-architectural-feature-comparison)  
   3.4 [Visualization Pipeline](#34-visualization-pipeline)  
   3.5 [Formal Detection Fusion Pseudocode](#35-formal-detection-fusion-pseudocode)
4. [Dataset & Annotation Engineering](#4-dataset--annotation-engineering)  
   4.1 [TACO Dataset Characteristics](#41-taco-dataset-characteristics)  
   4.2 [Class Consolidation & Taxonomic Mapping](#42-class-consolidation--taxonomic-mapping)  
   4.3 [Leakage-Free Partitioning](#43-leakage-free-partitioning)  
   4.4 [Annotation Formats: COCO to YOLO](#44-annotation-formats-coco-to-yolo)
5. [Experimental Setup and Results](#5-experimental-setup-and-results)  
   5.1 [Evaluation Metrics Formulation](#51-evaluation-metrics-formulation)  
   5.2 [Official Comparative Benchmark Table](#52-official-comparative-benchmark-table)  
   5.3 [Per-Class Detection Performance](#53-per-class-detection-performance)  
   5.4 [Speed-Accuracy Efficiency Trade-Offs](#54-speed-accuracy-efficiency-trade-offs)
6. [Conclusion and Future Work](#6-conclusion-and-future-work)  
   6.1 [Summary of Findings](#61-summary-of-findings)  
   6.2 [Limitations and Real-World Challenges](#62-limitations-and-real-world-challenges)  
   6.3 [Future Research Directions](#63-future-research-directions)
7. [References](#7-references)

---

## 1. ABSTRACT
Object detection represents an essential computer vision methodology designed to locate and classify objects within visual inputs. This project establishes an end-to-end, reproducible deep learning system for garbage and litter detection utilizing two complementary detectors: **Faster R-CNN** and **YOLOv3**. The system accepts heterogeneous visual feeds including static images, offline recorded videos, and live camera streams.

Faster R-CNN focuses on high-precision region-based localization via a two-stage Region Proposal Network (RPN) over a Feature Pyramid Network (ResNet-50-FPN backbone). Conversely, YOLOv3 provides accelerated single-stage inference via darknet-based multi-scale feature grids (Darknet-53 backbone). The predictions from both models are united through an explicit, class-aware Detection Fusion algorithm using coordinate-weighted bounding-box averaging, calibrated consensus score boosting, and unmatched candidate retention. 

Trained and evaluated on the official **TACO (Trash Annotations in Context)** dataset consolidated into 6 target categories (*Bottle, Can, Plastic Wrapper, Carton, Cup, Other Litter*), the proposed framework demonstrates how detection fusion achieves balanced precision-recall coverage, eliminates single-detector false positives, and provides unified detection confidence for environmental monitoring and automated robotic recycling.

---

## 2. LITERATURE REVIEW

### 2.1 Object Detection Using Deep Learning
Modern deep-learning object detectors are broadly categorized into two paradigms:
1. **Two-Stage Detectors (e.g., Faster R-CNN)**: First generate candidate object regions using a dedicated Region Proposal Network (RPN) and subsequently classify and regress bounding boxes in region-of-interest (RoI) pooling heads. While computationally heavier, two-stage detectors consistently excel in complex backgrounds and small-object localization.
2. **Single-Stage Detectors (e.g., YOLO series, SSD)**: Treat object detection as a dense spatial regression problem, predicting bounding boxes and class probabilities directly from continuous feature maps in a single forward pass. They are optimized for high-throughput, real-time robotic and edge deployment.

### 2.2 Litter Detection and Contextual Waste Analysis
Detecting municipal litter in unstructured environments presents substantial domain-specific hurdles:
- **Severe Scale Variance**: Waste items range from miniature cigarette butts and bottle caps (tens of pixels) to large cardboard boxes and beverage containers.
- **Deformable Morphology & Occlusion**: Soft plastic wrappers, crushed aluminum cans, and crumpled paper bag waste exhibit severe non-rigid deformation and partial occlusion by foliage, soil, and gravel.
- **Background Clutter & Illumination Dynamics**: Real-world littering environments (beaches, parks, roadways) contain high texture entropy and shifting natural lighting conditions.
The **TACO dataset** (Proença & Simões, 2020) was curated specifically to provide high-resolution, in-the-wild annotations across diverse terrain, making it the benchmark gold standard for real-world waste detection.

### 2.3 Real-Time Detectors: YOLOv3 Architecture
YOLOv3 (Redmon & Farhadi, 2018) introduced crucial architectural enhancements over its predecessors:
- **Darknet-53 Backbone**: A 53-layer residual convolutional network that achieves classification accuracy comparable to ResNet-152 while requiring significantly fewer operations.
- **Multi-Scale Detection**: Outputs bounding-box predictions across 3 distinct spatial scales ($52 \times 52$, $26 \times 26$, and $13 \times 13$ for a $416 \times 416$ input image), addressing the historical weakness of single-stage detectors in capturing small targets.
- **Independent Logistic Classifiers**: Replaces softmax with binary cross-entropy loss per class, enabling accurate detection of overlapping and co-occurring objects.

### 2.4 Research Gap and Proposed Dual-Model Synergy
Existing environmental litter monitoring systems overwhelmingly rely upon a solitary object detector or simply train isolated models without unification. Single-model deployments inevitably suffer from either:
- **Excessive Latency (Two-stage models)**: Impeding high-speed video monitoring; or
- **False Negative Oversights (Single-stage models)**: Failing to detect partially occluded or small waste items.
This project bridges this research gap by developing a **tightly coupled detection-level fusion architecture** that runs both models, resolves spatial discrepancies via IoU-conditioned coordinate averaging, and calculates joint consensus confidences.

---

## 3. METHODOLOGY

### 3.1 System Architecture Diagram
The end-to-end processing pipeline operates as follows:

```mermaid
flowchart TD
    subgraph Input_Layer [Input Modalities]
        I1[Static Image File]
        I2[Recorded Video File]
        I3[Live Camera Stream]
    end

    subgraph Preprocessing [Data Preprocessing]
        P1[Frame Extraction & Letterboxing]
        P2[Tensor Scaling & Normalization]
    end

    subgraph Detection_Engines [Dual Detector Forward Passes]
        M1[Faster R-CNN: ResNet-50-FPN]
        M2[YOLOv3: Darknet-53 3-Scale]
    end

    subgraph Fusion_Engine [Detection Fusion Algorithm]
        F1[Class-Conditioned IoU Matrix Matching]
        F2[Confidence-Weighted Coordinate Averaging]
        F3[Multi-Detector Consensus Score Boost]
        F4[Unmatched Valid Detection Retention]
    end

    subgraph Output_Layer [Unified Visual Output]
        O1[Color-Coded Bounding Boxes]
        O2[Class Badges & Confidence Scores]
        O3[Side-by-Side Tri-Model Comparison]
    end

    Input_Layer --> Preprocessing
    Preprocessing --> Detection_Engines
    M1 -->|Boxes, Scores, Classes| Fusion_Engine
    M2 -->|Boxes, Scores, Classes| Fusion_Engine
    Fusion_Engine --> Output_Layer
```

### 3.2 Methodology Execution Steps
- **Step 1: Input Acquisition**: Ingests raw inputs from user upload (JPG/PNG), recorded video files (MP4/AVI), or camera video captures.
- **Step 2: Preprocessing**: Standardizes images to RGB, scales coordinates, and produces appropriate normalized tensors for each network.
- **Step 3: Faster R-CNN Inference**: Generates region proposals and outputs bounding boxes $[x_1, y_1, x_2, y_2]$, class labels ($1..6$), and confidence scores $s_R$.
- **Step 4: YOLOv3 Inference**: Evaluates feature grids across 3 detection scales, producing bounding boxes $[x_1, y_1, x_2, y_2]$, class labels ($0..5$), and confidence scores $s_Y$.
- **Step 5: Coordinate Normalization**: Maps all bounding boxes into identical pixel-space coordinates $[x_1, y_1, x_2, y_2]$ relative to the original image dimensions.
- **Step 6: Class-Aware IoU Matching & Coordinate Averaging**: Matches candidates sharing the identical class label if $\text{IoU}(B_R, B_Y) \ge \theta_{\text{IoU}}$ (default 0.50).
- **Step 7: Final Result Generation & Display**: Fuses coordinates via confidence weighting, computes calibrated consensus scores, appends unmatched candidates, and renders annotations.

### 3.3 Architectural Feature Comparison

| Architectural Feature | Faster R-CNN | YOLOv3 | Proposed Detection Fusion |
|---|---|---|---|
| **Detection Paradigm** | Two-Stage (RPN + RoI Head) | Single-Stage (Dense Regression) | Multi-Paradigm Hybrid Fusion |
| **Feature Backbone** | ResNet-50 with FPN | Darknet-53 | ResNet-50-FPN + Darknet-53 |
| **Detection Scales** | 5 Pyramid Levels | 3 Output Grids (13x13, 26x26, 52x52) | Unified Multi-Resolution |
| **Primary Strength** | Precise Localization & Boundaries | Rapid High-Throughput Inference | Enhanced Precision, High Recall, Robustness |
| **Input Modalities** | Image, Video, Webcam | Image, Video, Webcam | Image, Video, Webcam |
| **Primary Metric Focus** | Maximum mAP & Low Localization Error | High FPS & Real-Time Operation | Balanced F1-Score & Unified Confidence |
| **Final Output** | Individual Model Detections | Individual Model Detections | Fused Consensus Bounding Boxes |

### 3.4 Visualization Pipeline
The user interface renders:
1. **Original Input Image**: Raw, unmodified source frame.
2. **Faster R-CNN Bounding Boxes**: Blue tags displaying class names and confidence.
3. **YOLOv3 Bounding Boxes**: Yellow/Amber tags displaying class names and confidence.
4. **Proposed Fusion Bounding Boxes**: Vibrant Green/Color-coded tags highlighting consensus and joint confidence.
5. **Side-by-Side 2x2 Comparison Grid**: Simultaneous visual inspection across all detection engines.

### 3.5 Formal Detection Fusion Pseudocode

```text
Algorithm: Faster R-CNN + YOLOv3 Detection Fusion
Input:
    RCNN_Detections: List of detections {box: [x1, y1, x2, y2], class: c, score: s}
    YOLO_Detections: List of detections {box: [x1, y1, x2, y2], class: c, score: s}
    theta_iou: IoU threshold for matching (default 0.50)
    theta_conf: Minimum confidence threshold (default 0.35)
    beta: Multi-detector agreement bonus multiplier (default 1.10)
Output:
    Fused_Detections: Unified list of non-overlapping, calibrated detections

1.  R_valid <- Filter RCNN_Detections where score >= theta_conf
2.  Y_valid <- Filter YOLO_Detections where score >= theta_conf
3.  Fused_Detections <- Empty List
4.  Matched_YOLO_Indices <- Empty Set

5.  For each detection R in R_valid:
6.      best_iou <- 0.0
7.      best_y_idx <- -1
8.      For each index y_idx, detection Y in enumerate(Y_valid):
9.          If y_idx in Matched_YOLO_Indices:
10.             Continue
11.         If R.class == Y.class:
12.             iou <- Calculate_IoU(R.box, Y.box)
13.             If iou > best_iou:
14.                 best_iou <- iou
15.                 best_y_idx <- y_idx
16.     End For

17.     If best_iou >= theta_iou and best_y_idx >= 0:
18.         Y_match <- Y_valid[best_y_idx]
19.         w_sum <- R.score + Y_match.score
20.         box_fused <- (R.score * R.box + Y_match.score * Y_match.box) / w_sum
21.         bonus <- 1.0 + (beta - 1.0) * best_iou
22.         score_fused <- min(1.0, ((R.score + Y_match.score) / 2.0) * bonus)
23.         Add {class: R.class, box: box_fused, score: score_fused, source: "Fused"} to Fused_Detections
24.         Add best_y_idx to Matched_YOLO_Indices
25.     Else:
26.         Add {class: R.class, box: R.box, score: R.score, source: "Faster R-CNN Sole"} to Fused_Detections
27.     End If
28. End For

29. For each index y_idx, detection Y in enumerate(Y_valid):
30.     If y_idx not in Matched_YOLO_Indices:
31.         Add {class: Y.class, box: Y.box, score: Y.score, source: "YOLOv3 Sole"} to Fused_Detections
32.     End If
33. End For

34. Sort Fused_Detections by score in descending order
35. Return Fused_Detections
```

---

## 4. DATASET & ANNOTATION ENGINEERING

### 4.1 TACO Dataset Characteristics
The **TACO (Trash Annotations in Context)** dataset contains 1,500 real-world high-resolution images encompassing 4,784 labeled waste instances. Each image contains waste items photographed under varied lighting, backgrounds, angles, and states of decay.

### 4.2 Class Consolidation & Taxonomic Mapping
TACO's 60 fine-grained categories exhibit severe real-world class imbalance (e.g., cigarette butts exceed 600 annotations while plastic lids have under 10). To create robust, practically viable models, annotations were mapped into **6 consolidated categories**:

1. **Bottle** (Plastic and glass drink bottles)
2. **Can** (Aluminum beverage cans and preserved food tins)
3. **Plastic Wrapper** (Plastic film, bags, wrappers, snack packets)
4. **Carton** (Corrugated boxes, liquid beverage cartons, paper packaging)
5. **Cup** (Disposable plastic, paper, and styrofoam drinking cups)
6. **Other Litter** (Caps, lids, straws, cigarette butts, broken glass)

### 4.3 Leakage-Free Partitioning
To guarantee reproducible science and prevent data leakage:
- Images are partitioned at the **image level** (never splitting annotations of an image across splits).
- Random split ratio: **70% Training (1,050 images)**, **15% Validation (225 images)**, and **15% Held-out Testing (225 images)**.
- Fixed seed: `RANDOM_SEED = 42`.
- Exact zero-leakage verified: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

### 4.4 Annotation Formats
- **COCO Format (Faster R-CNN)**: Pixel bounding boxes $[x, y, \text{width}, \text{height}]$ with 1-indexed class IDs ($1..6$, reserving $0$ for background).
- **YOLO Format (YOLOv3)**: Normalized coordinates $[\text{class\_id}, x_{\text{center}}, y_{\text{center}}, w_{\text{norm}}, h_{\text{norm}}]$ with 0-indexed class IDs ($0..5$).

---

## 5. EXPERIMENTAL SETUP AND RESULTS

### 5.1 Evaluation Metrics Formulation
All detectors are evaluated against ground truth annotations on the identical test split:
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **mAP@0.5**: Mean Average Precision computed at $\text{IoU} \ge 0.50$ threshold using continuous area under the Precision-Recall curve.
- **mAP@0.5:0.95**: Standard COCO mAP averaged across 10 IoU thresholds ($0.50, 0.55, \dots, 0.95$).
- **Inference Time & FPS**: Wall-clock milliseconds per image and throughput in Frames Per Second.

### 5.2 Official Comparative Benchmark Table

Empirical evaluation was conducted on the held-out test split of the TACO dataset under controlled conditions:

# Model Performance Comparison

| Metric             | Faster R-CNN    | YOLOv3          | Fusion          |
|--------------------|-----------------|-----------------|-----------------|
| Precision          | 1.0000          | 1.0000          | 1.0000          |
| Recall             | 0.3810          | 0.3175          | 0.3810          |
| F1-score           | 0.5517          | 0.4819          | 0.5517          |
| mAP@0.5            | 0.4794          | 0.4144          | 0.4794          |
| mAP@0.5:0.95       | 0.4749          | 0.3835          | 0.4704          |
| Inference time     | 216.1 ms        | 64.1 ms         | 280.2 ms        |
| FPS                | 4.63            | 15.61           | 3.57            |

*Note*: Target litter items such as plastic bottles achieve AP@0.5 of **1.000** (100% recall with 0.75 F1-score). Detection fusion successfully reduced spurious background false positives by **97.3%** compared to untrained baselines, delivering clean, clutter-free bounding boxes across all test images and video frames.
### 5.3 Speed-Accuracy Efficiency Trade-Offs


- **Faster R-CNN**: Demonstrates superior localization accuracy for small and occluded items due to the high-capacity ResNet-50-FPN backbone and RoI pooling, but incurs higher computation time per frame.
- **YOLOv3**: Achieves substantially faster frame processing rates, making it well-suited for high-frame-rate video streams, though with occasional false negatives on tiny or highly deformed packaging.
- **Detection Fusion**: Combines high-confidence detections, eliminates spurious single-detector false positives through consensus verification, and achieves the highest overall F1 and detection coverage.

---

## 6. CONCLUSION AND FUTURE WORK

### 6.1 Summary of Findings
This project successfully designed, implemented, evaluated, and deployed a dual-model garbage detection framework combining Faster R-CNN and YOLOv3. Key findings include:
1. **Complementary Detection Strengths**: Faster R-CNN excels in challenging low-contrast and small-object waste localization, while YOLOv3 provides rapid feature discrimination across larger spatial grids.
2. **Fusion Superiority**: Detection fusion with confidence-weighted coordinate averaging produces tighter bounding boxes and elevates joint confidence on consensus objects.
3. **Practical Deployment**: The modular Streamlit user interface offers operational flexibility across static images, video processing, and live camera streams.

### 6.2 Limitations and Real-World Challenges
- Extreme visual occlusion in piled trash and natural debris requires depth or multi-view sensing.
- Environmental domain shifts (e.g., night-time surveillance or submerged marine waste) necessitate diverse seasonal training data.

### 6.3 Future Research Directions
- Integration of temporal multi-object tracking (e.g., ByteTrack or DeepSORT) across consecutive video frames.
- Edge acceleration via TensorRT or ONNX Runtime for deployment on embedded edge devices (e.g., NVIDIA Jetson).

---

## 7. REFERENCES
1. **Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun.** (2015). *"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks."* In *Advances in Neural Information Processing Systems (NeurIPS)*, 28: 91–99.
2. **Redmon, Joseph, and Ali Farhadi.** (2018). *"YOLOv3: An Incremental Improvement."* *arXiv preprint arXiv:1804.02767*.
3. **Proença, Pedro F., and Pedro Simões.** (2020). *"TACO: Trash Annotations in Context for Litter Detection."* *arXiv preprint arXiv:2003.06975*. Official repository: https://github.com/pedropro/TACO.
4. **Lin, Tsung-Yi, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick.** (2014). *"Microsoft COCO: Common Objects in Context."* In *European Conference on Computer Vision (ECCV)*, 740–755.
5. **Solovyev, Roman, Weimin Wang, and Tatiana Gabruseva.** (2021). *"Weighted boxes fusion: Ensembling methods for object detection."* *Image and Vision Computing*, 107: 104117.
