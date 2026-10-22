"""
Streamlit Web Application for Garbage Detection and Classification.
Provides interactive inference across:
1. Static Images (Faster R-CNN, YOLOv3, and Proposed Detection Fusion)
2. Recorded Videos (frame-by-frame processing & annotated export)
3. Empirical Performance Evaluation (tables & plots)
4. Dataset Explorer (TACO categories & distribution)
"""

import sys
import time
import json
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

# Configure page
st.set_page_config(
    page_title="WasteAI - Garbage Detection & Fusion",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from models.faster_rcnn_detector import FasterRCNNDetector
from models.yolov3_detector import YOLOv3Detector
from fusion.detection_fusion import DetectionFusion
from utils.helpers import load_image, draw_detections, BenchmarkTimer

# Custom CSS for modern, clean dark-mode UI
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #58a6ff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .badge-waste {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detectors():
    """Initializes and caches model instances."""
    rcnn = FasterRCNNDetector(conf_thresh=0.20)
    yolo = YOLOv3Detector(conf_thresh=0.20)
    return rcnn, yolo

@st.cache_data
def get_unique_benchmark_samples():
    """Returns deduplicated, naturally sorted list of benchmark images."""
    img_dir = config.DATASET_DIR / "images" / "batch_1"
    unique_items = []
    seen = set()
    
    # Read files in sorted order
    if img_dir.exists():
        for p in sorted(img_dir.glob("*.jpg")):
            fname = p.name
            if fname.lower() not in seen:
                seen.add(fname.lower())
                unique_items.append((f"batch_1/{fname}", p))
    return unique_items

# Load scaled ground truth index
@st.cache_data
def load_benchmark_gt():
    scaled_gt_file = config.DATASET_DIR / "taco_annotations_scaled.json"
    if scaled_gt_file.exists():
        try:
            with open(scaled_gt_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

bench_gt_db = load_benchmark_gt()
benchmark_samples = get_unique_benchmark_samples()

# Sidebar Header & Branding
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 15px 0;">
    <div style="display: inline-block; width: 54px; height: 54px; border-radius: 12px; background: linear-gradient(135deg, #2ea043, #1f6feb); line-height: 54px; font-size: 28px;">
        ♻️
    </div>
    <h3 style="margin: 8px 0 2px 0; color: #58a6ff; font-weight: 700;">WasteAI</h3>
    <p style="margin: 0; color: #8b949e; font-size: 0.82rem;">Dual-Model Garbage Detection & Fusion</p>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "Navigation",
    [
        "Image Detection",
        "Video Detection",
        "Performance Evaluation",
        "Dataset Explorer",
        "Home & Overview"
    ],
    index=0
)

# Detection Parameters (Clean & Simple)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Parameters")

conf_slider = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.85,
    value=0.20,
    step=0.05,
    help="Minimum score required to display a detected litter object."
)

with st.sidebar.expander("Advanced Fusion Settings"):
    iou_slider = st.slider("IoU Fusion Threshold", 0.15, 0.70, 0.30, 0.05)
    bonus_slider = st.slider("Consensus Agreement Bonus", 1.0, 1.35, 1.20, 0.05)

st.sidebar.info(f"**Execution Hardware**: `{config.DEVICE.upper()}`")
st.sidebar.caption("TACO Benchmark Dataset (CC BY 4.0)")

# Initialize detectors & fusion engine
frcnn, yolo = load_detectors()
fusion = DetectionFusion(
    iou_threshold=iou_slider,
    faster_rcnn_conf_thresh=conf_slider,
    yolov3_conf_thresh=conf_slider,
    fused_conf_thresh=conf_slider,
    bonus_factor=bonus_slider
)

# ==============================================================================
# SECTION 1: IMAGE DETECTION (PRIMARY FOCUS)
# ==============================================================================
if nav == "Image Detection":
    st.title("🖼️ Garbage Detection & Dual-Model Comparison")
    st.markdown("Select any benchmark image or upload your own to detect litter using **Faster R-CNN**, **YOLOv3**, and **Proposed Detection Fusion**.")

    source_type = st.radio("Choose Input Mode:", ["Benchmark Sample Image", "Upload Custom Image"], horizontal=True)

    img_input = None
    selected_name = ""

    if source_type == "Benchmark Sample Image":
        if len(benchmark_samples) > 0:
            options_display = [f"[{i+1:02d}/80] {rel_name}" for i, (rel_name, _) in enumerate(benchmark_samples)]
            choice_idx = st.selectbox(
                "Select TACO Benchmark Image (80 Unique Samples):",
                range(len(options_display)),
                format_func=lambda i: options_display[i],
                index=0
            )
            selected_name = benchmark_samples[choice_idx][0]
            selected_path = benchmark_samples[choice_idx][1]
            img_input = Image.open(selected_path).convert("RGB")

            # Show Ground Truth Badge
            if selected_name in bench_gt_db:
                gt_objs = bench_gt_db[selected_name].get("objects", [])
                counts = Counter([o["class_name"] for o in gt_objs])
                gt_summary = ", ".join([f"**{cls}** ({cnt})" for cls, cnt in counts.items()])
                st.markdown(
                    f'<div style="background: rgba(46, 160, 67, 0.12); border-left: 4px solid #2ea043; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">'
                    f'🎯 <b>Verified Ground Truth:</b> {gt_summary} &nbsp;·&nbsp; <span style="color:#8b949e">{len(gt_objs)} total objects annotated</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No benchmark images found in dataset/images/batch_1.")
    else:
        uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img_input = Image.open(uploaded_file).convert("RGB")
            selected_name = uploaded_file.name

    if img_input:
        rgb_arr = np.array(img_input)

        run_btn = st.button("⚡ Run Dual Detection & Fusion", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Executing Faster R-CNN, YOLOv3, and Fusion Engine..."):
                rcnn_dets, rcnn_lat = frcnn.predict(rgb_arr, conf_thresh=conf_slider)
                yolo_dets, yolo_lat = yolo.predict(rgb_arr, conf_thresh=conf_slider)
                t_f0 = time.perf_counter()
                fused_dets = fusion.fuse(rcnn_dets, yolo_dets)
                f_lat = (time.perf_counter() - t_f0) * 1000.0 + (rcnn_lat + yolo_lat)

            # Performance Bar
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Faster R-CNN (Two-Stage)</div><div class="metric-value">{len(rcnn_dets)} <span style="font-size:0.9rem;color:#8b949e">objs ({rcnn_lat:.1f}ms)</span></div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">YOLOv3 (Single-Stage)</div><div class="metric-value">{len(yolo_dets)} <span style="font-size:0.9rem;color:#8b949e">objs ({yolo_lat:.1f}ms)</span></div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Proposed Detection Fusion</div><div class="metric-value" style="color:#3fb950">{len(fused_dets)} <span style="font-size:0.9rem;color:#8b949e">objs ({f_lat:.1f}ms)</span></div></div>', unsafe_allow_html=True)

            # Prominent Fused Detection Result
            st.subheader("🏆 Proposed Detection Fusion (Final Output)")
            p_fused = draw_detections(rgb_arr, fused_dets, title="Detection Fusion Result")
            st.image(p_fused, use_container_width=True)

            # Clean Detected Objects Summary Badges
            if len(fused_dets) > 0:
                st.markdown("#### Detected Objects:")
                badge_cols = st.columns(min(4, len(fused_dets)))
                for idx, d in enumerate(fused_dets):
                    cname = d["class_name"]
                    score = d["score"]
                    src = d.get("source", "Fused")
                    color_hex = config.CLASS_HEX_COLORS.get(cname, "#58a6ff")
                    with badge_cols[idx % len(badge_cols)]:
                        st.markdown(
                            f'<div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-left:4px solid {color_hex}; border-radius:8px; padding:8px 12px; margin-bottom:8px;">'
                            f'<b style="color:{color_hex}; font-size:1.05rem;">{cname}</b><br/>'
                            f'<span style="color:#c9d1d9; font-size:0.88rem;">Conf: <b>{score*100:.1f}%</b></span> · <span style="color:#8b949e; font-size:0.75rem;">{src}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Side-by-Side Model Comparison
            st.markdown("---")
            st.subheader("🔍 Model Comparison: Faster R-CNN vs. YOLOv3")
            col_rcnn, col_yolo = st.columns(2)
            with col_rcnn:
                st.markdown("##### Faster R-CNN (ResNet-50-FPN)")
                p_rcnn = draw_detections(rgb_arr, rcnn_dets, title="Faster R-CNN")
                st.image(p_rcnn, use_container_width=True)
            with col_yolo:
                st.markdown("##### YOLOv3 (Darknet-53)")
                p_yolo = draw_detections(rgb_arr, yolo_dets, title="YOLOv3")
                st.image(p_yolo, use_container_width=True)

            # Detections Inventory Table
            with st.expander("📋 View Detailed Bounding Box Inventory"):
                rows = []
                for idx, d in enumerate(fused_dets, 1):
                    b = d["box"]
                    rows.append({
                        "#": idx,
                        "Class": d["class_name"],
                        "Confidence": f"{d['score']*100:.1f}%",
                        "Source": d.get("source", "Fused"),
                        "Box [x1, y1, x2, y2]": f"[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]",
                        "Size (WxH)": f"{abs(b[2]-b[0]):.1f} x {abs(b[3]-b[1]):.1f} px"
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No litter detected at current threshold. Try lowering the threshold slider.")
        else:
            # Show original image preview before clicking run
            st.image(rgb_arr, caption=f"Selected Input: {selected_name}", use_container_width=True)

# ==============================================================================
# SECTION 2: VIDEO DETECTION
# ==============================================================================
elif nav == "Video Detection":
    st.title("🎥 Offline Video Detection & Analysis")
    st.markdown("Process video footage frame-by-frame using Faster R-CNN and YOLOv3.")

    video_source = st.radio("Select Video Input:", ["Demo Benchmark Video", "Upload Video File"], horizontal=True)
    tfile = None

    if video_source == "Demo Benchmark Video":
        demo_path = config.SAMPLES_DIR / "demo_litter_video.mp4"
        if demo_path.exists():
            tfile = demo_path
            st.info(f"Loaded Demo Video: `{demo_path.name}`")
            st.video(str(tfile))
        else:
            st.warning("Demo video not found. Please upload a video below.")
    else:
        video_file = st.file_uploader("Upload Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        if video_file:
            tfile = config.BASE_DIR / "outputs" / "temp_input_video.mp4"
            with open(tfile, "wb") as f:
                f.write(video_file.read())
            st.video(str(tfile))

    if tfile and st.button("⚡ Process Video Stream", type="primary"):
        cap = cv2.VideoCapture(str(tfile))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        max_frames_to_process = min(total_frames, 60)
        
        st_frame = st.empty()
        p_bar = st.progress(0)
        status_text = st.empty()

        processed = 0
        detected_counts = Counter()
        t_start = time.perf_counter()
        
        while cap.isOpened() and processed < max_frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r_d, _ = frcnn.predict(rgb, conf_thresh=conf_slider)
            y_d, _ = yolo.predict(rgb, conf_thresh=conf_slider)
            f_d = fusion.fuse(r_d, y_d)
            
            for d in f_d:
                detected_counts[d["class_name"]] += 1
                
            ann = draw_detections(rgb, f_d, title=f"Frame {processed+1}/{max_frames_to_process}")
            st_frame.image(ann, use_container_width=True)
            processed += 1
            p_bar.progress(processed / max_frames_to_process)
            status_text.text(f"Processing frame {processed}/{max_frames_to_process}...")
            
        cap.release()
        total_time = time.perf_counter() - t_start
        avg_fps = processed / max(0.001, total_time)
        status_text.empty()
        
        st.success(f"✅ Video analysis complete! Processed {processed} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")
        
        st.markdown("#### 📊 Cumulative Litter Inventory Detected in Video Stream:")
        if detected_counts:
            cols = st.columns(len(detected_counts))
            for idx, (cname, cnt) in enumerate(detected_counts.items()):
                color_hex = config.CLASS_HEX_COLORS.get(cname, "#58a6ff")
                with cols[idx]:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-left:4px solid {color_hex}; border-radius:8px; padding:10px 14px; text-align:center;">'
                        f'<div style="color:{color_hex}; font-weight:700; font-size:1.1rem;">{cname}</div>'
                        f'<div style="color:#c9d1d9; font-size:1.3rem; font-weight:700; margin-top:4px;">{cnt} <span style="font-size:0.8rem; font-weight:400; color:#8b949e">detections</span></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No objects detected above threshold.")

# ==============================================================================
# SECTION 3: PERFORMANCE EVALUATION
# ==============================================================================
elif nav == "Performance Evaluation":
    st.title("📊 Empirical Performance Evaluation & Benchmark")
    st.markdown("Official comparative metrics across held-out test images.")

    tbl_path = config.EVALUATIONS_DIR / "comparison_table.md"
    if tbl_path.exists():
        st.markdown(tbl_path.read_text(encoding="utf-8"))

    f_map = config.FIGURES_DIR / "model_map_comparison.png"
    if f_map.exists():
        st.subheader("mAP Comparison Across Models")
        st.image(str(f_map), use_container_width=True)

    f_lat = config.FIGURES_DIR / "latency_vs_map.png"
    if f_lat.exists():
        st.subheader("Speed vs. Accuracy Trade-Off (Latency vs mAP)")
        st.image(str(f_lat), use_container_width=True)

# ==============================================================================
# SECTION 4: DATASET EXPLORER
# ==============================================================================
elif nav == "Dataset Explorer":
    st.title("🔍 TACO Dataset Explorer")
    st.markdown("Explore the 6 consolidated waste categories from the TACO dataset.")

    st.subheader("Target Categories & Color Coding")
    cols = st.columns(6)
    for idx, cname in enumerate(config.TARGET_CLASSES):
        hex_c = config.CLASS_HEX_COLORS.get(cname, "#58a6ff")
        with cols[idx]:
            st.markdown(
                f'<div style="background:{hex_c}; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:700;">'
                f'{cname}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.subheader("Benchmark Sample Gallery")
    if benchmark_samples:
        gcols = st.columns(3)
        for idx, (rel_name, p) in enumerate(benchmark_samples[:9]):
            with gcols[idx % 3]:
                st.image(str(p), caption=rel_name, use_container_width=True)

# ==============================================================================
# SECTION 5: HOME & OVERVIEW
# ==============================================================================
elif nav == "Home & Overview":
    st.title("♻️ Garbage Detection and Classification Using Faster R-CNN and YOLOv3")
    st.markdown("### A Deep-Learning Dual-Model Fusion Framework for Environmental Litter Monitoring")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-title">Framework</div><div class="metric-value">PyTorch 2.x</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-title">Dataset</div><div class="metric-value">TACO Benchmark</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-title">Target Classes</div><div class="metric-value">6 Categories</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-title">Inference Engine</div><div class="metric-value">Dual Fusion</div></div>', unsafe_allow_html=True)

    st.markdown("""
    #### Architectural Summary
    - **Faster R-CNN (Two-Stage Detector)**: Employs a Region Proposal Network (RPN) over a ResNet-50-FPN backbone for superior localization accuracy.
    - **YOLOv3 (Single-Stage Detector)**: Fast spatial regression across 3 multi-scale feature grids (Darknet-53 backbone).
    - **Proposed Detection Fusion**: Combines detections through class-conditioned IoU matching, confidence-weighted coordinate averaging, and calibrated consensus score boosting.
    """)
