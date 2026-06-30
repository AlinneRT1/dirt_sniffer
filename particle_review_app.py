"""
Unified Particle Detection Gallery
All-in-one: summary table + gallery + mass edit + full image zoom/pan
KEY: Click particles to zoom into full image with pan controls

Usage:
    streamlit run particle_review_gallery_unified.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
from copy import deepcopy
import plotly.graph_objects as go
import plotly.express as px

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

st.set_page_config(page_title="Particle Detection Review", page_icon="icon.ico", layout="wide")
st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" width="50">
        <h1 style="margin: 0; padding: 0;">dirt_sniffer: Review Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

st.divider()

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
if "selected_for_fullview" not in st.session_state:
    st.session_state.selected_for_fullview = None

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
                st.rerun()
    
    st.divider()
    
    # Undo
    if st.session_state.undo_stack:
        if st.button("↶ Undo"):
            st.session_state.results = st.session_state.undo_stack.pop()
            st.session_state.selected_particles = set()
            st.rerun()
    
    # Stats
    if st.session_state.results:
        total = sum(len([p for p in ps if not p["deleted"]]) for ps in st.session_state.results.values())
        black_count = sum(len([p for p in ps if p["black_bg"] and not p["deleted"]]) for ps in st.session_state.results.values())
        
        st.success(f"✅ {len(st.session_state.results)} images")
        st.info(f"📊 {total} particles")
        if black_count > 0:
            st.warning(f"⚫ {black_count} black bg")
        
        st.write(f"**Selected:** {len(st.session_state.selected_particles)}")
    
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
        
        if rows:
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download",
                csv,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN: UNIFIED GALLERY
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.results:
    st.info("👈 Upload images and run inference")
else:
    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("📊 Summary Table")
    
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
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=150)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: FILTERS & GALLERY GRID
    # ─────────────────────────────────────────────────────────────────────────
    
    st.subheader("🖼️ Particle Gallery & Editor")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_cls = st.selectbox("Class:", ["Fiber", "Glass", "Metallic", "Other"], key="sel_cls")
    with col2:
        sel_bin = st.selectbox("Bin:", [b[0] for b in SIZE_BINS], key="sel_bin")
    with col3:
        show_selected_only = st.checkbox("Show selected only")
    
    # Collect matching
    matching = []
    for img_name, ps in st.session_state.results.items():
        for idx, p in enumerate(ps):
            key = f"{img_name}_{idx}"
            if p["class"] == sel_cls and p["size_bin"] == sel_bin and not p["deleted"]:
                if not show_selected_only or key in st.session_state.selected_particles:
                    matching.append({"img": img_name, "idx": idx, "particle": p, "key": key})
    
    if matching:
        st.success(f"{len(matching)} particles")
        
        # Gallery grid (5 columns for better view)
        cols = st.columns(5)
        for i, match in enumerate(matching):
            with cols[i % 5]:
                img_name = match["img"]
                pidx = match["idx"]
                p = match["particle"]
                key = match["key"]
                
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
                    
                    # Highlight selected
                    is_selected = key in st.session_state.selected_particles
                    border_color = (0, 255, 0) if is_selected else (100, 100, 100)
                    
                    # Show image
                    st.image(crop, use_column_width=True, 
                            caption=f"{p['diameter_um']}µm")
                    
                    # Checkbox to select
                    if st.checkbox("📌 Select", value=is_selected, key=f"sel_{key}"):
                        st.session_state.selected_particles.add(key)
                    else:
                        st.session_state.selected_particles.discard(key)
                    
                    st.caption(f"{p['class']}\n{p['confidence']:.2f}")
                    
                    # Quick action: zoom to full image
                    if st.button("🔍 View Full", key=f"view_{key}"):
                        st.session_state.selected_for_fullview = (img_name, pidx)
                        st.rerun()
                    
                    if p["black_bg"]:
                        st.warning("⚫ Black BG", icon="⚠️")
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────────────
        # PART 3: MASS EDIT IN GALLERY
        # ─────────────────────────────────────────────────────────────────────
        
        if st.session_state.selected_particles:
            st.subheader("⚙️ Bulk Edit Selected Particles")
            
            selected_count = len(st.session_state.selected_particles)
            st.info(f"**{selected_count} particle(s) selected**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                action = st.radio("Action:", ["Delete All Selected", "Change Class To"])
            
            with col2:
                if action == "Change Class To":
                    new_cls = st.selectbox("New class:", ["Fiber", "Glass", "Metallic", "Other"])
            
            if st.button("🔥 Execute Action"):
                push_undo()
                for key in st.session_state.selected_particles:
                    img_name, idx = key.rsplit("_", 1)
                    idx = int(idx)
                    if action == "Delete All Selected":
                        st.session_state.results[img_name][idx]["deleted"] = True
                    else:
                        st.session_state.results[img_name][idx]["class"] = new_cls
                        st.session_state.results[img_name][idx]["size_bin"] = get_size_bin(
                            st.session_state.results[img_name][idx]["diameter_um"]
                        )
                st.session_state.selected_particles = set()
                st.success(f"✅ Applied to {selected_count} particles")
                st.rerun()
    
    else:
        st.info(f"No particles for {sel_cls} in {sel_bin}")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PART 4: FULL IMAGE ZOOM/PAN VIEW (when particle is clicked)
    # ─────────────────────────────────────────────────────────────────────────
    
    if st.session_state.selected_for_fullview:
        st.subheader("🔎 Full Image View - Zoom & Pan")
        
        img_name, pidx = st.session_state.selected_for_fullview
        f = st.session_state.uploaded_files_cache.get(img_name)
        
        if f:
            img = Image.open(f)
            img_np = np.array(img)
            
            p = st.session_state.results[img_name][pidx]
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            
            # Create Plotly figure with image
            fig = go.Figure()
            
            # Add image as background
            fig.add_trace(go.Image(
                z=img_np,
                name="Image"
            ))
            
            # Highlight the particle with a rectangle
            fig.add_shape(
                type="rect",
                x0=x, y0=y, x1=x+w, y1=y+h,
                line=dict(color="lime", width=3),
                name="Particle"
            )
            
            fig.update_layout(
                title=f"Image: {img_name} | Particle: {p['class']} ({p['diameter_um']}µm)",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                height=700,
            )
            
            # Enable zoom/pan
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Controls
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Class:** {p['class']}")
            with col2:
                st.write(f"**Size:** {p['diameter_um']}µm")
            with col3:
                if st.button("✕ Close Full View"):
                    st.session_state.selected_for_fullview = None
                    st.rerun()
