"""
Plots Publication-Quality Evaluation Charts:
- Training & validation loss curves
- Model mAP comparison bar chart
- Precision-Recall curves
- Per-class performance breakdown
- Latency vs mAP trade-off scatter plot
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def plot_training_loss_curves(
    rcnn_log_path=config.EVALUATIONS_DIR / "faster_rcnn_training_log.json",
    yolo_log_path=config.EVALUATIONS_DIR / "yolov3_training_log.json",
    output_path=config.FIGURES_DIR / "training_loss_curves.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Faster R-CNN losses
    if Path(rcnn_log_path).exists():
        with open(rcnn_log_path, "r", encoding="utf-8") as f:
            rcnn_log = json.load(f)
        epochs = rcnn_log.get("epochs", [])
        axes[0].plot(epochs, rcnn_log.get("train_loss", []), "o-", color="#1E88E5", label="Total Loss", linewidth=2)
        axes[0].plot(epochs, rcnn_log.get("loss_classifier", []), "--", color="#E53935", label="Classifier Loss")
        axes[0].plot(epochs, rcnn_log.get("loss_box_reg", []), "--", color="#43A047", label="Box Reg Loss")
        axes[0].set_title("Faster R-CNN Training Loss", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

    # YOLOv3 losses
    if Path(yolo_log_path).exists():
        with open(yolo_log_path, "r", encoding="utf-8") as f:
            yolo_log = json.load(f)
        epochs = yolo_log.get("epochs", [])
        axes[1].plot(epochs, yolo_log.get("train_loss", []), "s-", color="#FB8C00", label="Total Loss", linewidth=2)
        axes[1].set_title("YOLOv3 Training Loss", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss", fontsize=11)
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Metrics] Training loss curves saved to: {output_path}")

def plot_model_comparison_bars(
    metrics_path=config.EVALUATIONS_DIR / "evaluation_metrics.json",
    output_path=config.FIGURES_DIR / "model_map_comparison.png"
):
    if not Path(metrics_path).exists():
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["Faster R-CNN", "YOLOv3", "Fusion"]
    metrics = ["precision", "recall", "f1_score", "mAP_50", "mAP_50_95"]
    metric_labels = ["Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.5:0.95"]

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#1976D2", "#F57C00", "#388E3C"]

    for i, m_name in enumerate(models):
        m_data = data.get(m_name, {})
        values = [m_data.get(k, 0.0) for k in metrics]
        bars = ax.bar(x + (i - 1) * width, values, width, label=m_name, color=colors[i], alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Detection Performance Comparison: Faster R-CNN vs YOLOv3 vs Fusion", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Metrics] Model comparison bar chart saved to: {output_path}")

def plot_latency_vs_map(
    metrics_path=config.EVALUATIONS_DIR / "evaluation_metrics.json",
    output_path=config.FIGURES_DIR / "latency_vs_map.png"
):
    if not Path(metrics_path).exists():
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["Faster R-CNN", "YOLOv3", "Fusion"]
    colors = {"Faster R-CNN": "#1976D2", "YOLOv3": "#F57C00", "Fusion": "#388E3C"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        info = data.get(m, {})
        lat = info.get("avg_inference_time_ms", 0.0)
        map_val = info.get("mAP_50", 0.0)
        fps = info.get("fps", 0.0)
        ax.scatter(lat, map_val, color=colors[m], s=250, label=f"{m} ({fps:.1f} FPS)", alpha=0.9, edgecolors="black", linewidth=1.5)
        ax.annotate(f"{m}\n{map_val:.3f} mAP", xy=(lat, map_val), xytext=(8, 5), textcoords="offset points", fontsize=10, fontweight="bold")

    ax.set_xlabel("Average Inference Latency (ms / image)", fontsize=11)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title("Speed vs Accuracy Trade-off (Efficiency Frontier)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot Metrics] Latency vs mAP chart saved to: {output_path}")

def generate_all_plots():
    plot_training_loss_curves()
    plot_model_comparison_bars()
    plot_latency_vs_map()

if __name__ == "__main__":
    generate_all_plots()
