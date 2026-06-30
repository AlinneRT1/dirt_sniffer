"""
Unified Particle Detection Gallery - Streamlit Fixed Version
✅ Works with any resolution without dependencies like cv2 or pi-heif issues
✅ Fixed permission errors for Ultralytics settings directory

Usage:
    streamlit run particle_review_gallery_fixed_streamlit.py
"""

import os
import tempfile
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import streamlit as st
from ultralytics import YOLO
from copy import deepcopy
import base64
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Use a writable directory for Ultralytics to avoid permission denied errors
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

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

# Streamlit page setup
st.set_page_config(page_title="Particle Detection Review", page_icon="icon.ico", layout="wide")

# Load logo icon and embed in the UI
with open("icon.png", "rb") as f:
    img = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div style="display:flex;align-items:center;gap:15px;">
    <img src="data:image/png;base64,{img}" width="80">
    <h1 style="margin:0;">🧹 dirt_sniffer: Review Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load YOLO model for particle detection."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}.")
        return None
    return YOLO(MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_size_bin(diameter_um):
    """Categorize particle size into bins."""
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"


def is_black_background(image_np, x, y, w, h, threshold=BLACK_BG_THRESHOLD):
    """Detect areas with black background."""
    try:
        region = image_np[max(0, y - 5):min(image_np.shape[0], y + h + 5),
                          max(0, x - 5):min(image_np.shape[1], x + w + 5)]
        return np.mean(region) < threshold if region.size > 0 else False
    except Exception as e:
        print(f"Error detecting black background: {str(e)}")
        return False


def process_image(image_path, model):
    """
    Run YOLO inference on the image. Uses PIL for image processing with no fixed limits.
    """
    try:
        img_pil = Image.open(image_path)

        # Convert to RGB format if the original is not RGB
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        # Convert to numpy array for processing
        image = np.array(img_pil)

        if image is None or image.size == 0:
            st.error(f"❌ Empty or corrupted image: {image_path}")
            return None

        results = model(image, iou=0.45, conf=0.02, verbose=False)

        particles = []
        for r in results:
            if r.boxes and r.masks:
                for mask, box, cls, conf in zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    label = model.names[int(cls)]

                    box_w = x2 - x1
                    box_h = y2 - y1
                    max_diam_um = max(box_w, box_h) * CALIBRATION_UM_PER_PIXEL

                    is_black = is_black_background(image, x1, y1, box_w, box_h)

                    particles.append({
                        "x": x1,
                        "y": y1,
                        "w": box_w,
                        "h": box_h,
                        "class": label,
                        "confidence": round(conf, 3),
                        "diameter_um": round(max_diam_um, 1),
                        "size_bin": get_size_bin(max_diam_um),
                        "deleted": False,
                        "black_bg": is_black
                    })

        return particles if particles else None
    except Exception as e:
        st.error(f"❌ Error processing {image_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = {}
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "selected_particles" not in st.session_state:
    st.session_state.selected_particles = set()
if "uploaded_files_cache" not in st.session_state:
    st.session_state.uploaded_files_cache = {}

# Undo functionality
def push_undo():
    """Save the current state for undo."""
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - UPLOAD & CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload & Process")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
        help="Upload high-resolution images for particle detection."
    )

    # Run inference on uploaded files
    if uploaded_files and st.button("🔍 Run Inference"):
        model = load_model()
        if model:
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, file in enumerate(uploaded_files):
                    temp_path = os.path.join(tmpdir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    particles = process_image(temp_path, model)
                    if particles:
                        st.session_state.results[file.name] = particles
                        st.session_state.uploaded_files_cache[file.name] = file
                st.success("✅ Inference completed!")