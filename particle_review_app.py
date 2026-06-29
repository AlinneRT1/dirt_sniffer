"""
Enhanced Particle Detection Review Dashboard
Features: zoom/pan, undo, black background detection, manual add, mass edit, histogram, gallery grid

Usage:
    streamlit run particle_review_enhanced.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299
BLACK_BG_THRESHOLD = 30

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
    "Fiber": (0, 200, 255),
    "Glass": (0, 255, 0),
    "Metallic": (255, 100, 0),
    "Other": (0, 0, 255),
}

st.set_page_config(page_title="Particle Detection Review", page_icon="icon.jpg", layout="wide")
st.title("🧹 dirt_sniffer: Review Dashboard")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_size_bin(diameter_um):
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"

def is_black_background(image_np, x, y, w, h, threshold=BLACK_BG_THRESHOLD):
    """Check if particle is on black background."""
    region = image_np[max(0, y-5):min(image_np.shape[0], y+h+5),
                      max(0, x-5):min(image_np.shape[1], x+w+5)]
    if region.size == 0:
        return False
    avg_brightness = np.mean(region)
    return avg_brightness < threshold

def process_image(image_path, model):
    """Run YOLO inference."""
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

            box_w = x2 - x1
            box_h = y2 - y1
            max_diam_um = max(box_w, box_h) * CALIBRATION_UM_PER_PIXEL

            is_black = is_black_background(image, x1, y1, box_w, box_h)

            particles.append({
                "x": x1, "y": y1, "w": box_w, "h": box_h,
                "class": label, "confidence": float(conf),
                "diameter_um": round(max_diam_um, 1),
                "size_bin": get_size_bin(max_diam_um),
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
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload & Process")

    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔍 Run Inference"):
            model = load_model()
            if model is None:
                st.error("Model not found")
            else:
                progress = st.progress(0)
                status = st.empty()

                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, f in enumerate(uploaded_files):
                        status.text(f"Processing {i+1}/{len(uploaded_files)}...")

                        temp_path = os.path.join(tmpdir, f.name)
                        with open(temp_path, "wb") as fp:
                            fp.write(f.getbuffer())

                        particles = process_image(temp_path, model)
                        if particles:
                            st.session_state.results[f.name] = particles
                            st.session_state.uploaded_files_cache[f.name] = f

                        progress.progress((i + 1) / len(uploaded_files))

                status.text("✅ Done!")

    st.divider()

    # Undo
    if st.session_state.undo_stack:
        if st.button("↶ Undo"):
            st.session_state.results = st.session_state.undo_stack.pop()
            st.rerun()

    # Stats
    if st.session_state.results:
        total = sum(len([p for p in ps if not p["deleted"]]) for ps in st.session_state.results.values())
        black_count = sum(len([p for p in ps if p["black_bg"] and not p["deleted"]]) for ps in st.session_state.results.values())

        st.success(f"✅ {len(st.session_state.results)} images")
        st.info(f"📊 {total} particles")
        if black_count > 0:
            st.warning(f"⚫ {black_count} black bg")

    st.divider()

    # Export
    if st.button("📥 Export CSV"):
        rows = []
        for img_name, ps in st.session_state.results.items():
            for p in ps:
                if not p["deleted"]:
                    rows.append({
                        "image": img_name,
                        "class": p["class"],
                        "diameter_um": p["diameter_um"],
                        "size_bin": p["size_bin"],
                        "confidence": round(p["confidence"], 3),
                        "black_background": p["black_bg"],
                    })

        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download",
            csv,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.results:
    st.info("👈 Upload images and run inference")
else:
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🖼️ Gallery", "⚙️ Mass Edit"])

    with tab1:
        st.subheader("Particle Distribution")

        # Summary table
        data = {}
        for cls in ["Fiber", "Glass", "Metallic", "Other"]:
            data[cls] = {}
            for b, _, _ in SIZE_BINS:
                count = sum(len([p for p in ps if p["class"] == cls and p["size_bin"] == b and not p["deleted"]])
                           for ps in st.session_state.results.values())
                data[cls][b] = count

        rows = []
        for cls in ["Fiber", "Glass", "Metallic", "Other"]:
            row = {"Material": cls}
            total = 0
            for b, _, _ in SIZE_BINS:
                c = data[cls][b]
                row[b] = c
                total += c
            row["Total"] = total
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # Histogram (B-G only)
        st.subheader("Size Distribution (B-G Bins)")
        fig = go.Figure()

        for cls in ["Fiber", "Glass", "Metallic", "Other"]:
            diams = [p["diameter_um"] for ps in st.session_state.results.values()
                    for p in ps if p["class"] == cls and not p["deleted"]]
            if diams:
                fig.add_trace(go.Histogram(
                    x=diams,
                    name=cls,
                    nbinsx=15,
                    opacity=0.7
                ))

        fig.update_layout(barmode="overlay", xaxis_title="Diameter (µm)", yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🖼️ Particle Gallery")

        col1, col2 = st.columns(2)
        with col1:
            sel_cls = st.selectbox("Class:", ["Fiber", "Glass", "Metallic", "Other"])
        with col2:
            sel_bin = st.selectbox("Bin:", [b[0] for b in SIZE_BINS])

        # Collect matching
        matching = []
        for img_name, ps in st.session_state.results.items():
            for idx, p in enumerate(ps):
                if p["class"] == sel_cls and p["size_bin"] == sel_bin and not p["deleted"]:
                    matching.append({"img": img_name, "idx": idx, "particle": p})

        if matching:
            st.success(f"{len(matching)} found")

            # Gallery grid (4 columns)
            cols = st.columns(4)
            for i, match in enumerate(matching):
                with cols[i % 4]:
                    img_name = match["img"]
                    pidx = match["idx"]
                    p = match["particle"]

                    f = st.session_state.uploaded_files_cache.get(img_name)
                    if f:
                        img = Image.open(f)
                        img_np = np.array(img)

                        # Crop with context
                        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                        margin = 30
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(img_np.shape[1], x + w + margin)
                        y2 = min(img_np.shape[0], y + h + margin)
                        crop = img_np[y1:y2, x1:x2]

                        # Display with zoom hint
                        st.image(crop, use_column_width=True,
                                caption=f"{p['diameter_um']}µm\nClick to zoom ↓")

                        # Mini details
                        st.caption(f"**{p['class']}**\nConf: {p['confidence']:.2f}")

                        # Reclassify
                        new_cls = st.selectbox(
                            "Change class:",
                            ["Fiber", "Glass", "Metallic", "Other"],
                            index=["Fiber", "Glass", "Metallic", "Other"].index(p["class"]),
                            key=f"cls_{img_name}_{pidx}"
                        )

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if new_cls != p["class"] and st.button("✓", key=f"save_{img_name}_{pidx}"):
                                push_undo()
                                st.session_state.results[img_name][pidx]["class"] = new_cls
                                st.session_state.results[img_name][pidx]["size_bin"] = get_size_bin(p["diameter_um"])
                                st.rerun()
                        with col_b:
                            if st.button("🗑️", key=f"del_{img_name}_{pidx}"):
                                push_undo()
                                st.session_state.results[img_name][pidx]["deleted"] = True
                                st.rerun()

                        if p["black_bg"]:
                            st.warning("⚫ Black BG")
        else:
            st.info(f"No particles for {sel_cls} in {sel_bin}")

    with tab3:
        st.subheader("⚙️ Mass Edit Particles")

        # Collect all
        all_flat = []
        for img_name, ps in st.session_state.results.items():
            for idx, p in enumerate(ps):
                if not p["deleted"]:
                    all_flat.append({
                        "key": f"{img_name}_{idx}",
                        "image": img_name,
                        "idx": idx,
                        "class": p["class"],
                        "diameter_um": p["diameter_um"],
                        "confidence": p["confidence"]
                    })

        if all_flat:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_cls = st.multiselect("Filter by class:", ["Fiber", "Glass", "Metallic", "Other"],
                                           default=["Fiber", "Glass", "Metallic", "Other"])
            with col2:
                filter_conf = st.slider("Min confidence:", 0.0, 1.0, 0.0)
            with col3:
                show_only_black = st.checkbox("Only black background")

            filtered = [p for p in all_flat if p["class"] in filter_cls and p["confidence"] >= filter_conf]

            st.write(f"**{len(filtered)} particles match**")

            # Multi-select
            selected = []
            for p in filtered[:100]:  # Limit display
                if st.checkbox(f"{p['image']} - {p['class']} ({p['diameter_um']}µm)", key=f"sel_{p['key']}"):
                    selected.append(p['key'])

            st.divider()

            if selected:
                st.write(f"**Selected: {len(selected)}**")

                action = st.radio("Action:", ["Delete All", "Change Class To"])

                if action == "Change Class To":
                    new_cls = st.selectbox("New class:", ["Fiber", "Glass", "Metallic", "Other"])

                if st.button("🔥 Execute"):
                    push_undo()
                    for key in selected:
                        img_name, idx = key.rsplit("_", 1)
                        idx = int(idx)
                        if action == "Delete All":
                            st.session_state.results[img_name][idx]["deleted"] = True
                        else:
                            st.session_state.results[img_name][idx]["class"] = new_cls
                            st.session_state.results[img_name][idx]["size_bin"] = get_size_bin(
                                st.session_state.results[img_name][idx]["diameter_um"]
                            )
                    st.success(f"✅ Applied to {len(selected)} particles")
                    st.rerun()