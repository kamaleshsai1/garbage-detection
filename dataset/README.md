# TACO: Trash Annotations in Context for Litter Detection

## Overview
This project uses the **TACO (Trash Annotations in Context)** dataset for training, evaluating, and demonstrating deep-learning models (Faster R-CNN, YOLOv3, and Detection Fusion) on garbage and litter classification.

- **Official Website & Repository**: [https://github.com/pedropro/TACO](https://github.com/pedropro/TACO)
- **Zenodo Record**: [https://zenodo.org/records/3587843](https://zenodo.org/records/3587843)
- **Paper**: Pedro F. Proença and Pedro Simões. *"TACO: Trash Annotations in Context for Litter Detection"*, arXiv:2003.06975, 2020.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## Annotation Format & Hierarchical Taxonomy
TACO provides annotations in the standard **COCO (Common Objects in Context)** format:
- `images`: Image ID, file name, width, height, Flickr URLs (original and resized 640px).
- `annotations`: Annotation ID, image ID, category ID, bounding box `[x, y, width, height]`, area, and segmentation polygon.
- `categories`: 60 fine-grained categories organized under 28 supercategories.

### Class Consolidation & Taxonomy Mapping
The 60 fine-grained categories have severe class imbalance in real-world settings (e.g., individual counts ranging from over 600 cigarette butts down to 1 plastic lid). To ensure robust model training and meaningful practical evaluation, categories are consolidated into **6 well-represented municipal litter classes**:

| Target Class ID | Target Class Name | Original TACO Supercategories / Categories Included |
|---|---|---|
| **0** (YOLO) / **1** (R-CNN) | **Bottle** | Clear plastic bottle, Glass bottle, Other plastic bottle |
| **1** (YOLO) / **2** (R-CNN) | **Can** | Drink can, Food can, Aerosol |
| **2** (YOLO) / **3** (R-CNN) | **Plastic Wrapper** | Plastic film, Other plastic wrapper, Crisp packet, Plastic bag |
| **3** (YOLO) / **4** (R-CNN) | **Carton** | Drink carton, Corrugated carton, Meal carton, Egg carton, Paper |
| **4** (YOLO) / **5** (R-CNN) | **Cup** | Disposable plastic cup, Paper cup, Coffee cup |
| **5** (YOLO) / **6** (R-CNN) | **Other Litter** | Bottle cap, Cigarette butt, Straw, Styrofoam piece, Pop tab, Broken glass |

*Note: For Faster R-CNN, Class 0 is reserved for Background, shifting class IDs to 1-6.*

---

## Dataset Splits & Reproducibility
- **Split Ratio**: 70% Training, 15% Validation, 15% Held-out Testing.
- **No Image Leakage**: Splitting is strictly performed at the image level using fixed seed `RANDOM_SEED = 42`.
- **Integrity**: Missing image URLs or download timeouts are handled gracefully without fabricating annotations.
