"""
Particle Detection Review Dashboard
A Streamlit app to upload images, run YOLO inference, review and validate results.

Usage:
    pip install streamlit ultralytics opencv-python pillow pandas numpy
    streamlit run particle_review_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from ultralytics import YOLO
import json
import tempfile
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299

SIZE_BINS = [
    ("B: 5-15um (1519 pcs)", 5, 15),
    ("C: 12-25um (186 pcs)", 15, 25),
    ("D: 25-50um (67 pcs)", 25, 50),
    ("E: 50-100um (9 pcs)", 50, 100),
    ("F: 100-250um (1 pcs)", 100, 250),
    ("G: 250-500um (0 pcs)", 250, 500),
    ("H: 500-750um (0 pcs)", 500, 750),
    ("I: 750-100um (0 pcs)", 750, 1000),
    ("J: 1000um+ (0 pcs)", 1000, float("inf")),
]

CLASS_COLORS = {
    "Fiber": (0, 200, 255),      # cyan
    "Glass": (0, 255, 0),        # green
    "Metallic": (255, 100, 0),   # blue
    "Other": (0, 0, 255),        # red
}


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Particle Detection Review", layout="wide")
st.title("🔬 Dirt Sniffer 🧹 Review Dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def get_size_bin(diameter_um):
    """Return size bin label for diameter in µm."""
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"


def process_image(image_path, model):
    """Run YOLO inference on image and extract particle data."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    results = model(image, iou=0.45, conf=0.02, verbose=False)
    
    particles = []
    for r in results:
        if r.boxes is None or r.masks is None:
            continue

        for mask, box, cls, conf in zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label = model.names[int(cls)]
            
            # Calculate size from bounding box
            box_w = x2 - x1
            box_h = y2 - y1
            max_diam_px = max(box_w, box_h)
            max_diam_um = max_diam_px * CALIBRATION_UM_PER_PIXEL
            
            size_bin = get_size_bin(max_diam_um)
            
            particles.append({
                "x": x1,
                "y": y1,
                "w": box_w,
                "h": box_h,
                "class": label,
                "confidence": float(conf),
                "diameter_um": round(max_diam_um, 1),
                "size_bin": size_bin,
                "mask": mask.astype(np.int32),
                "deleted": False
            })
    
    return particles


def draw_particle(image, particle, color=(255, 255, 255), thickness=2):
    """Draw a single particle on image."""
    x, y, w, h = particle["x"], particle["y"], particle["w"], particle["h"]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = {}  # {image_name: particles list}
if "current_view" not in st.session_state:
    st.session_state.current_view = "dashboard"
if "selected_class" not in st.session_state:
    st.session_state.selected_class = None
if "selected_bin" not in st.session_state:
    st.session_state.selected_bin = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload & Process")
    
    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🔍 Run Inference", key="run_inference"):
            model = load_model()
            if model is None:
                st.error("Could not load model!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {i+1}/{len(uploaded_files)}...")
                        
                        # Save to temp
                        temp_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Infer
                        particles = process_image(temp_path, model)
                        if particles:
                            st.session_state.results[uploaded_file.name] = particles
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text(f"✅ Done! Processed {len(st.session_state.results)} images.")
    
    st.divider()
    
    if st.session_state.results:
        st.success(f"✅ {len(st.session_state.results)} images processed")
        total_particles = sum(
            len([p for p in particles if not p["deleted"]])
            for particles in st.session_state.results.values()
        )
        st.info(f"📊 Total particles: {total_particles}")
    
    st.divider()
    st.header("💾 Export")
    if st.button("📥 Export Results as CSV"):
        rows = []
        for image_name, particles in st.session_state.results.items():
            for i, p in enumerate(particles):
                if not p["deleted"]:
                    rows.append({
                        "image": image_name,
                        "particle_id": i,
                        "class": p["class"],
                        "diameter_um": p["diameter_um"],
                        "size_bin": p["size_bin"],
                        "confidence": round(p["confidence"], 3),
                        "x": p["x"],
                        "y": p["y"],
                    })
        
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"particle_results_{timestamp}.csv",
            mime="text/csv"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: DASHBOARD VIEW
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.results:
    st.info("👈 Upload images and run inference to get started!")
else:
    # Build summary table
    summary_data = {}
    for class_name in ["Fiber", "Glass", "Metallic", "Other"]:
        summary_data[class_name] = {}
        for bin_label, _, _ in SIZE_BINS:
            count = 0
            for particles in st.session_state.results.values():
                count += len([
                    p for p in particles 
                    if p["class"] == class_name and p["size_bin"] == bin_label and not p["deleted"]
                ])
            summary_data[class_name][bin_label] = count
    
    # Display as table
    st.subheader("📊 Particle Distribution by Class & Size")
    
    table_data = []
    for class_name in ["Fiber", "Glass", "Metallic", "Other"]:
        row = {"Material": class_name}
        total = 0
        for bin_label, _, _ in SIZE_BINS:
            count = summary_data[class_name][bin_label]
            row[bin_label] = count
            total += count
        row["Total"] = total
        table_data.append(row)
    
    df_summary = pd.DataFrame(table_data)
    
    # Display with styling
    st.dataframe(
        df_summary,
        use_container_width=True,
        height=200
    )
    
    st.divider()
    
    # Click to view gallery
    st.subheader("🖼️ Review Particles by Class & Size")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.selectbox(
            "Select Class:",
            ["Fiber", "Glass", "Metallic", "Other"],
            key="class_select"
        )
    with col2:
        selected_bin = st.selectbox(
            "Select Size Bin:",
            [b[0] for b in SIZE_BINS],
            key="bin_select"
        )
    
    # Collect matching particles
    matching_particles = []
    for image_name, particles in st.session_state.results.items():
        for idx, p in enumerate(particles):
            if (p["class"] == selected_class and 
                p["size_bin"] == selected_bin and 
                not p["deleted"]):
                matching_particles.append({
                    "image_name": image_name,
                    "particle_idx": idx,
                    "particle": p
                })
    
    if matching_particles:
        st.success(f"Found {len(matching_particles)} particle(s)")
        
        # Gallery
        cols = st.columns(4)
        for i, match in enumerate(matching_particles):
            with cols[i % 4]:
                image_name = match["image_name"]
                particle = match["particle"]
                
                # Load and crop image
                image_files = [f for f in uploaded_files if f.name == image_name]
                if image_files:
                    img = Image.open(image_files[0])
                    img_np = np.array(img)
                    
                    # Crop around particle
                    x, y, w, h = particle["x"], particle["y"], particle["w"], particle["h"]
                    margin = 10
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(img_np.shape[1], x + w + margin)
                    y2 = min(img_np.shape[0], y + h + margin)
                    crop = img_np[y1:y2, x1:x2]
                    
                    st.image(crop, use_column_width=True)
                    st.caption(f"{particle['class']}\n{particle['diameter_um']}µm")
                    
                    # Delete button
                    if st.button(
                        "🗑️ Delete",
                        key=f"del_{image_name}_{match['particle_idx']}"
                    ):
                        st.session_state.results[image_name][match['particle_idx']]["deleted"] = True
                        st.rerun()
    else:
        st.info(f"No particles found for {selected_class} in bin {selected_bin}")
