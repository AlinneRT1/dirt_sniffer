import streamlit as st
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

st.title("🧹 Particle Detection")

file = st.file_uploader("Upload image", type=["jpg", "png", "tif"])

if file:
    st.write("✓ File uploaded")
    img = Image.open(file)
    st.write(f"Image size: {img.size}")
    arr = np.array(img)
    st.write(f"Array shape: {arr.shape}")