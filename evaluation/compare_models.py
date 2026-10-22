"""
Model Comparison and Tabulation Module.
Generates the comparison table required by prompt requirement 7.
"""

import sys
import json
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def generate_comparison_table(metrics_file=config.EVALUATIONS_DIR / "evaluation_metrics.json"):
    metrics_file = Path(metrics_file)
    if not metrics_file.exists():
        print(f"Metrics file {metrics_file} not found. Run evaluate_models.py first.")
        return None

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rcnn = data.get("Faster R-CNN", {})
    yolo = data.get("YOLOv3", {})
    fusion = data.get("Fusion", {})

    metrics_list = [
        ("Precision", f"{rcnn.get('precision', 0.0):.4f}", f"{yolo.get('precision', 0.0):.4f}", f"{fusion.get('precision', 0.0):.4f}"),
        ("Recall", f"{rcnn.get('recall', 0.0):.4f}", f"{yolo.get('recall', 0.0):.4f}", f"{fusion.get('recall', 0.0):.4f}"),
        ("F1-score", f"{rcnn.get('f1_score', 0.0):.4f}", f"{yolo.get('f1_score', 0.0):.4f}", f"{fusion.get('f1_score', 0.0):.4f}"),
        ("mAP@0.5", f"{rcnn.get('mAP_50', 0.0):.4f}", f"{yolo.get('mAP_50', 0.0):.4f}", f"{fusion.get('mAP_50', 0.0):.4f}"),
        ("mAP@0.5:0.95", f"{rcnn.get('mAP_50_95', 0.0):.4f}", f"{yolo.get('mAP_50_95', 0.0):.4f}", f"{fusion.get('mAP_50_95', 0.0):.4f}"),
        ("Inference time", f"{rcnn.get('avg_inference_time_ms', 0.0):.1f} ms", f"{yolo.get('avg_inference_time_ms', 0.0):.1f} ms", f"{fusion.get('avg_inference_time_ms', 0.0):.1f} ms"),
        ("FPS", f"{rcnn.get('fps', 0.0):.2f}", f"{yolo.get('fps', 0.0):.2f}", f"{fusion.get('fps', 0.0):.2f}")
    ]

    header = f"| {'Metric':<18} | {'Faster R-CNN':<15} | {'YOLOv3':<15} | {'Fusion':<15} |"
    separator = f"|{'-'*20}|{'-'*17}|{'-'*17}|{'-'*17}|"
    rows = [header, separator]

    for m_name, r_val, y_val, f_val in metrics_list:
        rows.append(f"| {m_name:<18} | {r_val:<15} | {y_val:<15} | {f_val:<15} |")

    table_str = "\n".join(rows)
    print("\n=== EXPERIMENTAL EVALUATION COMPARISON TABLE ===")
    print(table_str)
    print("================================================\n")

    # Save to markdown report snippet
    out_md = config.EVALUATIONS_DIR / "comparison_table.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Model Performance Comparison\n\n" + table_str + "\n")

    return table_str

if __name__ == "__main__":
    generate_comparison_table()
