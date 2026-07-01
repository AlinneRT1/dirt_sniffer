import sys
from unittest.mock import MagicMock

sys.modules['pi_heif'] = MagicMock()

import os
import traceback

try:
    import streamlit as st
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    import warnings

    warnings.filterwarnings('ignore')

    st.set_page_config(page_title="Particle Detection", layout="wide")
    st.title("🧹 Particle Detection")

    if "model" not in st.session_state:
        st.session_state.model = None
    if "results" not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        st.header("Upload & Process")
        file = st.file_uploader("Image", type=["jpg", "png", "tif"])

        if file:
            if st.button("Run Inference"):
                status = st.status("Processing...", expanded=True)

                try:
                    with status:
                        st.write("📂 Loading image...")
                        img = Image.open(file)
                        st.write(f"✓ Image opened: {img.size}")

                        st.write("🔄 Converting to RGB...")
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        st.write("✓ RGB conversion done")

                        st.write("📥 Importing YOLO...")
                        from ultralytics import YOLO

                        st.write("✓ YOLO import successful")

                        st.write("⬇️ Loading model...")
                        if st.session_state.model is None:
                            st.session_state.model = YOLO("yolov8n-seg.pt")
                        st.write("✓ Model loaded")

                        model = st.session_state.model

                        st.write("🔍 Running inference...")
                        st.write(f"Input: {img.size} pixels")

                        results = model(img, conf=0.02, verbose=False)
                        st.write(f"✓ Inference complete")

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

                        st.session_state.results[file.name] = particles
                        st.write(f"✓ {len(particles)} particles detected!")

                    st.success("Done!")

                except Exception as e:
                    st.error(f"❌ Error: {type(e).__name__}")
                    st.error(str(e))
                    st.code(traceback.format_exc())

    if st.session_state.results:
        st.subheader("Results")
        for img_name, particles in st.session_state.results.items():
            st.write(f"**{img_name}**: {len(particles)} particles")

except Exception as e:
    st.error(f"❌ FATAL: {type(e).__name__}")
    st.error(str(e))
    st.code(traceback.format_exc())