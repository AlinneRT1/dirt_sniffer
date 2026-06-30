import os

os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'

import sys
from unittest.mock import MagicMock

sys.modules['pi_heif'] = MagicMock()

import streamlit as st
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import numpy as np
import tempfile
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Particle Detection", layout="wide")
st.title("🧹 dirt_sniffer: Particle Detection")

try:
    from ultralytics import YOLO

    model = None
except Exception as e:
    st.error(f"YOLO error: {e}")
    model = None

if "results" not in st.session_state:
    st.session_state.results = {}
if "cache" not in st.session_state:
    st.session_state.cache = {}

with st.sidebar:
    st.header("Upload & Process")
    files = st.file_uploader("Images", type=["jpg", "png", "tif"])

    if files and st.button("Run"):
        if model is None:
            st.error("Model not loaded")
        else:
            try:
                st.write("Loading image...")
                img = Image.open(files)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                arr = np.array(img)

                st.write(f"Image: {arr.shape}")
                st.write("Running YOLO...")

                results = model(arr, conf=0.02, verbose=False)

                particles = []
                for r in results:
                    if r.boxes is not None:
                        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                            particles.append({
                                "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                                "class": model.names[int(cls)],
                                "conf": float(conf),
                            })

                st.session_state.results[files.name] = particles
                st.session_state.cache[files.name] = files
                st.write(f"✓ {len(particles)} particles found")
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback

                st.error(traceback.format_exc())

if st.session_state.results:
    st.subheader("Results")
    total = sum(len(p) for p in st.session_state.results.values())
    st.metric("Total Particles", total)

    st.divider()
    st.write("Gallery (basic):")

    for img_name, particles in st.session_state.results.items():
        st.write(f"**{img_name}**: {len(particles)} particles")