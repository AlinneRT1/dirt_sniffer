"""
Particle Detection Gallery - Clean Version
"""
import os
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

import streamlit as st
import base64
import numpy as np
from PIL import Image
import pandas as pd
import tempfile
from datetime import datetime
from copy import deepcopy
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_OK = True
except:
    YOLO_OK = False

st.set_page_config(page_title="Particle Detection", page_icon="icon.png", layout="wide")

# Header
try:
    with open("icon.png", "rb") as f:
        img = base64.b64encode(f.read()).decode()
    st.markdown(f'<div style="display:flex;align-items:center;gap:15px;"><img src="data:image/png;base64,{img}" width="80"><h1 style="margin:0;">🧹 dirt_sniffer</h1></div>', unsafe_allow_html=True)
except:
    st.markdown("# 🧹 dirt_sniffer")

# CONFIG
MODEL_PATH = "yolov8n-seg.pt"  # Use pre-trained (auto-downloads)
TILE_SIZE = 3000
CALIBRATION = 1.299

SIZE_BINS = [
    ("B: 5-15μm", 5, 15),
    ("C: 12-25μm", 15, 25),
    ("D: 25-50μm", 25, 50),
    ("E: 50-100μm", 50, 100),
]

def get_size_bin(d):
    for label, lo, hi in SIZE_BINS:
        if lo <= d < hi:
            return label
    return "Other"

@st.cache_resource
def load_model():
    if not YOLO_OK:
        return None
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def process_image(path, model):
    try:
        st.write(f"📂 Loading {path}...")
        img = Image.open(path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_arr = np.array(img)
        h, w = img_arr.shape[:2]

        st.write(f"✓ Loaded {w}×{h}")

        st.write(f"🔍 Running inference...")
        results = model(img_arr, iou=0.45, conf=0.02, verbose=False)

        particles = []
        for r in results:
            if r.boxes is None or r.masks is None:
                continue

            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                w_px = x2 - x1
                h_px = y2 - y1
                d_um = max(w_px, h_px) * CALIBRATION

                particles.append({
                    "x": x1, "y": y1, "w": w_px, "h": h_px,
                    "class": model.names[int(cls)],
                    "confidence": float(conf),
                    "diameter_um": round(d_um, 1),
                    "size_bin": get_size_bin(d_um),
                    "deleted": False,
                })

        st.write(f"✓ Found {len(particles)} particles")
        return particles

    except Exception as e:
        st.error(f"❌ Processing error: {type(e).__name__}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# SESSION STATE
if "results" not in st.session_state:
    st.session_state.results = {}
if "cache" not in st.session_state:
    st.session_state.cache = {}

st.divider()

# SIDEBAR
with st.sidebar:
    st.header("📤 Upload & Process")

    files = st.file_uploader("Upload images", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True)

    if files and st.button("🔍 Run Inference"):
        model = load_model()
        if model is None:
            st.error("YOLO not available")
        else:
            progress = st.progress(0)
            for i, f in enumerate(files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f.name) as tmp:
                    tmp.write(f.getbuffer())
                    particles = process_image(tmp.name, model)
                    if particles:
                        st.session_state.results[f.name] = particles
                        st.session_state.cache[f.name] = f
                progress.progress((i + 1) / len(files))
            st.rerun()

# MAIN
if not st.session_state.results:
    st.info("👈 Upload images")
else:
    st.subheader("📊 Results")

    total = sum(len([p for p in ps if not p["deleted"]]) for ps in st.session_state.results.values())
    st.metric("Total Particles", total)

    st.divider()
    st.subheader("🖼️ Gallery")

    all_p = []
    for img_name, ps in st.session_state.results.items():
        for idx, p in enumerate(ps):
            if not p["deleted"]:
                all_p.append((f"{img_name}_{idx}", img_name, idx, p))

    if all_p:
        cols = st.columns(6)
        for i, (key, img_name, idx, p) in enumerate(all_p):
            with cols[i % 6]:
                f = st.session_state.cache.get(img_name)
                if f:
                    img = Image.open(f)
                    img_arr = np.array(img)

                    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                    crop = img_arr[max(0,y-15):min(img_arr.shape[0],y+h+15), max(0,x-15):min(img_arr.shape[1],x+w+15)]

                    st.image(crop, use_column_width=True, caption=f"{p['diameter_um']}µm")
                    st.caption(f"{p['class']}")

                    if st.button("🗑️ Delete", key=f"del_{key}"):
                        st.session_state.results[img_name][idx]["deleted"] = True
                        st.rerun()

    st.divider()

    # EXPORT
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
                    })

        if rows:
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download", csv, "results.csv", "text/csv")