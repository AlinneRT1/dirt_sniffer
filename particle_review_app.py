# CRITICAL: Mock cv2 BEFORE any imports that use it
import sys
from unittest.mock import MagicMock

sys.modules['cv2'] = MagicMock()
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
                        st.write(f"Input type: {type(img)}")
                        st.write(f"Input size: {img.size}")

                        try:
                            st.write("Starting model prediction...")
                            results = model(img, conf=0.02, verbose=False)
                            st.write(f"✓ Prediction returned: {type(results)}")
                            st.write(f"Results length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
                        except Exception as pred_err:
                            st.error(f"Prediction error: {pred_err}")
                            raise

                        st.write("✓ Inference complete")

                        particles = []
                        st.write("Processing results...")
                        for i, r in enumerate(results):
                            st.write(f"Result {i}: type={type(r)}")
                            if r.boxes is not None:
                                st.write(f"  Boxes: {len(r.boxes)}")
                                for j, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
                                    if j % 100 == 0:
                                        st.write(f"  Processing box {j}...")
                                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                                    particles.append({
                                        "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                                        "class": model.names[int(cls)],
                                        "conf": float(conf),
                                    })
                            else:
                                st.write(f"  No boxes in result {i}")

                        st.session_state.results[file.name] = particles
                        st.write(f"✓ {len(particles)} particles detected!")

                    st.success("Done!")

                except Exception as e:
                    st.error(f"❌ Error: {type(e).__name__}: {str(e)}")
                    st.code(traceback.format_exc())

    if st.session_state.results:
        st.subheader("Results")
        for img_name, particles in st.session_state.results.items():
            st.write(f"**{img_name}**: {len(particles)} particles")

except Exception as e:
    st.error(f"❌ FATAL: {type(e).__name__}: {str(e)}")
    st.code(traceback.format_exc())