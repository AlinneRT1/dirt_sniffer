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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from copy import deepcopy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299

SIZE_BINS = [
    ("B: 5-15μm (1519 pcs)", 5, 15),
    ("C: 12-25μm (186 pcs)", 15, 25),
    ("D: 25-50μm (67 pcs)", 25, 50),
    ("E: 50-100μm (9 pcs)", 50, 100),
    ("F: 100-250μm (1 pcs)", 100, 250),
    ("G: 250-500μm (0 pcs)", 250, 500),
    ("H: 500-750μm (0 pcs)", 500, 750),
    ("I: 750-100μm (0 pcs)", 750, 1000),
    ("J: 1000μm+ (0 pcs)", 1000, float("inf")),
]

CLASS_COLORS = {
    "Fiber": (0, 200, 255),      # cyan
    "Glass": (0, 255, 0),        # green
    "Metallic": (255, 100, 0),   # blue
    "Other": (0, 0, 255),        # red
}

BLACK_BG_THRESHOLD = 30  # pixels with avg value < 30 are considered "black"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def convert_tiff_to_rgb(image_path):
    """Convert TIFF (or any format) to RGB numpy array."""
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def is_black_background(image_np, x, y, w, h, threshold=BLACK_BG_THRESHOLD):
    """Check if particle is mostly on black background."""
    region = image_np[max(0, y - 5):min(image_np.shape[0], y + h + 5),
    max(0, x - 5):min(image_np.shape[1], x + w + 5)]
    if region.size == 0:
        return False
    avg_brightness = np.mean(region)
    return avg_brightness < threshold


def get_size_bin(diameter_um):
    """Return size bin label for diameter in µm."""
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"


def process_image(image_path, model):
    """Run YOLO inference on image and extract particle data."""
    image = convert_tiff_to_rgb(image_path)
    h, w = image.shape[:2]

    # Convert to BGR for YOLO
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2BGR)

    results = model(image_bgr, iou=0.45, conf=0.02, verbose=False)

    particles = []
    for r in results:
        if r.boxes is None or r.masks is None:
            continue

        for mask, box, cls, conf in zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label = model.names[int(cls)]

            box_w = x2 - x1
            box_h = y2 - y1
            max_diam_px = max(box_w, box_h)
            max_diam_um = max_diam_px * CALIBRATION_UM_PER_PIXEL

            size_bin = get_size_bin(max_diam_um)

            # Check for black background
            is_black = is_black_background(image, x1, y1, box_w, box_h)

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
                "deleted": False,
                "black_bg": is_black
            })

    return particles


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = {}
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "selected_particles" not in st.session_state:
    st.session_state.selected_particles = set()
if "uploaded_files_cache" not in st.session_state:
    st.session_state.uploaded_files_cache = {}


def push_undo():
    """Save current state to undo stack."""
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))


def pop_undo():
    """Restore previous state from undo stack."""
    if st.session_state.undo_stack:
        st.session_state.results = st.session_state.undo_stack.pop()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: UPLOAD & CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload & Process")

    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
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
                        status_text.text(f"Processing {i + 1}/{len(uploaded_files)}...")

                        temp_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        particles = process_image(temp_path, model)
                        if particles:
                            st.session_state.results[uploaded_file.name] = particles
                            st.session_state.uploaded_files_cache[uploaded_file.name] = uploaded_file

                        progress_bar.progress((i + 1) / len(uploaded_files))

                    status_text.text(f"✅ Done! Processed {len(st.session_state.results)} images.")

    st.divider()

    # Undo button
    if st.session_state.undo_stack:
        if st.button("↶ Undo Last Change"):
            pop_undo()

    # Statistics
    if st.session_state.results:
        total_particles = sum(
            len([p for p in particles if not p["deleted"]])
            for particles in st.session_state.results.values()
        )
        black_bg_count = sum(
            len([p for p in particles if p["black_bg"] and not p["deleted"]])
            for particles in st.session_state.results.values()
        )
        st.success(f"✅ {len(st.session_state.results)} images")
        st.info(f"📊 Total particles: {total_particles}")
        if black_bg_count > 0:
            st.warning(f"⚫ Black background: {black_bg_count}")

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
                        "black_background": p["black_bg"],
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
# MAIN: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.results:
    st.info("👈 Upload images and run inference to get started!")
else:
    # Build summary
    summary_data = {}
    for class_name in ["Fiber", "Glass", "Metallic", "Other"]:
        summary_data[class_name] = {}
        for bin_label, _, _ in SIZE_BINS:
            count = sum(
                len([p for p in particles if p["class"] == class_name and
                     p["size_bin"] == bin_label and not p["deleted"]])
                for particles in st.session_state.results.values()
            )
            summary_data[class_name][bin_label] = count

    # Tab 1: Dashboard with Histogram
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🖼️ Gallery", "⚙️ Mass Edit"])

    with tab1:
        st.subheader("Particle Distribution by Class & Size")

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
        st.dataframe(df_summary, use_container_width=True, height=200)

        st.divider()

        # Histogram
        st.subheader("📈 Size Distribution by Class")

        fig = go.Figure()

        for class_name in ["Fiber", "Glass", "Metallic", "Other"]:
            diameters = []
            for particles in st.session_state.results.values():
                for p in particles:
                    if p["class"] == class_name and not p["deleted"]:
                        diameters.append(p["diameter_um"])

            if diameters:
                fig.add_trace(go.Histogram(
                    x=diameters,
                    name=class_name,
                    nbinsx=20,
                    opacity=0.7
                ))

        fig.update_layout(
            barmode="overlay",
            xaxis_title="Diameter (µm)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🖼️ Review & Edit Particles")

        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox("Select Class:", ["Fiber", "Glass", "Metallic", "Other"], key="class_select")
        with col2:
            selected_bin = st.selectbox("Select Size Bin:", [b[0] for b in SIZE_BINS], key="bin_select")

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

            # Gallery with zoom, pan, reclassify, delete
            for i, match in enumerate(matching_particles):
                with st.container():
                    image_name = match["image_name"]
                    particle = match["particle"]

                    # Load image
                    uploaded_file = st.session_state.uploaded_files_cache.get(image_name)
                    if uploaded_file:
                        img = Image.open(uploaded_file)
                        img_np = np.array(img)

                        # Crop
                        x, y, w, h = particle["x"], particle["y"], particle["w"], particle["h"]
                        margin = 20
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(img_np.shape[1], x + w + margin)
                        y2 = min(img_np.shape[0], y + h + margin)
                        crop = img_np[y1:y2, x1:x2]

                        cols = st.columns([2, 1])
                        with cols[0]:
                            st.image(crop, use_column_width=True, caption=f"Particle {i + 1}")
                        with cols[1]:
                            st.write(f"**Class:** {particle['class']}")
                            st.write(f"**Size:** {particle['diameter_um']}µm")
                            st.write(f"**Conf:** {particle['confidence']:.2f}")
                            if particle["black_bg"]:
                                st.warning("⚫ Black BG")

                            # Reclassify
                            new_class = st.selectbox(
                                "Reclassify to:",
                                ["Fiber", "Glass", "Metallic", "Other"],
                                index=["Fiber", "Glass", "Metallic", "Other"].index(particle["class"]),
                                key=f"reclass_{image_name}_{match['particle_idx']}"
                            )

                            if new_class != particle["class"]:
                                if st.button("✓ Confirm", key=f"confirm_{image_name}_{match['particle_idx']}"):
                                    push_undo()
                                    st.session_state.results[image_name][match["particle_idx"]]["class"] = new_class
                                    st.session_state.results[image_name][match["particle_idx"]][
                                        "size_bin"] = get_size_bin(particle["diameter_um"])
                                    st.rerun()

                            # Delete
                            if st.button("🗑️ Delete", key=f"del_{image_name}_{match['particle_idx']}"):
                                push_undo()
                                st.session_state.results[image_name][match['particle_idx']]["deleted"] = True
                                st.rerun()

                    st.divider()
        else:
            st.info(f"No particles found for {selected_class} in bin {selected_bin}")

    with tab3:
        st.subheader("⚙️ Mass Edit")

        # Collect all particles for multi-select
        all_particles_flat = []
        for image_name, particles in st.session_state.results.items():
            for idx, p in enumerate(particles):
                if not p["deleted"]:
                    all_particles_flat.append({
                        "key": f"{image_name}_{idx}",
                        "image": image_name,
                        "idx": idx,
                        "class": p["class"],
                        "diameter_um": p["diameter_um"],
                        "confidence": p["confidence"]
                    })

        if all_particles_flat:
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_class = st.multiselect("Filter by class:", ["Fiber", "Glass", "Metallic", "Other"],
                                              default=["Fiber", "Glass", "Metallic", "Other"])
            with col2:
                filter_conf = st.slider("Min confidence:", 0.0, 1.0, 0.0)
            with col3:
                filter_black = st.checkbox("Show only black background")

            filtered = [
                p for p in all_particles_flat
                if p["class"] in filter_class and p["confidence"] >= filter_conf
            ]

            # Multi-select checkboxes
            st.write(f"**{len(filtered)} particles match filters**")
            selected_keys = []
            for p in filtered[:50]:  # Limit to 50 for performance
                if st.checkbox(f"{p['image']} - {p['class']} ({p['diameter_um']}µm)", key=f"select_{p['key']}"):
                    selected_keys.append(p['key'])

            st.divider()

            if selected_keys:
                st.write(f"**Selected: {len(selected_keys)} particles**")

                col1, col2 = st.columns(2)
                with col1:
                    mass_action = st.selectbox("Action:", ["Delete All", "Change Class To"])

                with col2:
                    if mass_action == "Change Class To":
                        new_class_mass = st.selectbox("New class:", ["Fiber", "Glass", "Metallic", "Other"])

                if st.button("🔥 Execute Action"):
                    push_undo()
                    for key in selected_keys:
                        image_name, idx = key.rsplit("_", 1)
                        idx = int(idx)
                        if mass_action == "Delete All":
                            st.session_state.results[image_name][idx]["deleted"] = True
                        else:
                            st.session_state.results[image_name][idx]["class"] = new_class_mass
                            st.session_state.results[image_name][idx]["size_bin"] = get_size_bin(
                                st.session_state.results[image_name][idx]["diameter_um"]
                            )
                    st.success(f"✅ Applied to {len(selected_keys)} particles")
                    st.rerun()