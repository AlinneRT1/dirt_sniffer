"""
Unified Particle Detection Gallery
All-in-one: summary table + gallery + mass edit + full image zoom/pan
KEY: Click particles to zoom into full image with pan controls

UPDATED: Sizing from mask bounds instead of bbox

Usage:
    streamlit run particle_review_gallery_unified.py
"""

import streamlit as st
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
import base64


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

with open("icon.png", "rb") as f:
    img = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div style="display:flex;align-items:center;gap:15px;">
    <img src="data:image/png;base64,{img}" width="80">
    <h1 style="margin:0;">🧹dirt_sniffer: Review Dashboard</h1>
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
    region = image_np[max(0, y - 5):min(image_np.shape[0], y + h + 5),
    max(0, x - 5):min(image_np.shape[1], x + w + 5)]
    if region.size == 0:
        return False
    avg_brightness = np.mean(region)
    return avg_brightness < threshold


def resize_image_for_display(image_array, max_height=1080):
    """Resize image to max height for faster display"""
    h, w = image_array.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        image_array = cv2.resize(image_array, (new_w, max_height))
    return image_array


def process_image(image_path, model):
    """Run YOLO inference - SIZE FROM MASK BOUNDS instead of bbox"""
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    results = model(image, iou=0.45, conf=0.02, verbose=False)

    particles = []
    for r in results:
        if r.boxes is None or r.masks is None:
            continue

        for i, (mask, box, cls, conf) in enumerate(zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label = model.names[int(cls)]

            box_w = x2 - x1
            box_h = y2 - y1

            # Initialize mask bounds variables
            x_min = x_max = y_min = y_max = None
            size_method = "bbox"
            max_diam_um = max(box_w, box_h) * CALIBRATION_UM_PER_PIXEL

            # Try to get mask bounds (for visualization and sizing)
            try:
                mask_data = r.masks.data[i]
                if hasattr(mask_data, 'cpu'):
                    mask_array = mask_data.cpu().numpy()
                else:
                    mask_array = mask_data

                # Find actual mask pixels
                mask_pixels = np.where(mask_array > 0.5)

                if len(mask_pixels[0]) > 0:
                    # Get tight bounds from mask
                    y_min, y_max = int(mask_pixels[0].min()), int(mask_pixels[0].max())
                    x_min, x_max = int(mask_pixels[1].min()), int(mask_pixels[1].max())

                    # Diameter from mask bounds (more accurate!)
                    mask_w = x_max - x_min + 1
                    mask_h = y_max - y_min + 1
                    max_diam_um = max(mask_w, mask_h) * CALIBRATION_UM_PER_PIXEL
                    size_method = "mask_bounds"
            except Exception as e:
                # If mask extraction fails, keep bbox sizing
                pass

            is_black = is_black_background(image, x1, y1, box_w, box_h)

            particles.append({
                "x": x1, "y": y1, "w": box_w, "h": box_h,
                "class": label, "confidence": float(conf),
                "diameter_um": round(max_diam_um, 1),
                "size_bin": get_size_bin(max_diam_um),
                "size_method": size_method,
                "mask_x_min": x_min,
                "mask_y_min": y_min,
                "mask_x_max": x_max,
                "mask_y_max": y_max,
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
                        status.text(f"Processing {i + 1}/{len(uploaded_files)}...")

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

    if st.session_state.undo_stack:
        if st.button("↶ Undo"):
            st.session_state.results = st.session_state.undo_stack.pop()
            st.session_state.selected_particles = set()
            st.rerun()

    if st.session_state.results:
        total = sum(len([p for p in ps if not p["deleted"]]) for ps in st.session_state.results.values())
        black_count = sum(
            len([p for p in ps if p["black_bg"] and not p["deleted"]]) for ps in st.session_state.results.values())

        st.success(f"✅ {len(st.session_state.results)} images")
        st.info(f"📊 {total} particles")
        if black_count > 0:
            st.warning(f"⚫ {black_count} black bg")

        st.write(f"**Selected:** {len(st.session_state.selected_particles)}")

    st.divider()

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
                        "size_method": p.get("size_method", "unknown"),
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
    # PART 2: FILTERS & ALL PARTICLE GALLERY
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("🖼️ All Particles Gallery")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_class = st.multiselect("Filter by class:", ["Fiber", "Glass", "Metallic", "Other"],
                                      default=["Fiber", "Glass", "Metallic", "Other"], key="fc")
    with col2:
        filter_bin = st.multiselect("Filter by size bin:", [b[0] for b in SIZE_BINS],
                                    default=[b[0] for b in SIZE_BINS], key="fb")
    with col3:
        show_black_only = st.checkbox("Black bg only")
    with col4:
        items_per_page = st.selectbox("Per page:", [12, 20, 36, 50], index=0)

    # Collect all particles
    all_particles = []
    for img_name, ps in st.session_state.results.items():
        for idx, p in enumerate(ps):
            if not p["deleted"]:
                key = f"{img_name}_{idx}"
                if (p["class"] in filter_class and
                        p["size_bin"] in filter_bin and
                        (not show_black_only or p["black_bg"])):
                    all_particles.append({
                        "key": key,
                        "img": img_name,
                        "idx": idx,
                        "particle": p
                    })

    if all_particles:
        st.success(f"{len(all_particles)} particles found")

        # Pagination
        total_pages = max(1, (len(all_particles) + items_per_page - 1) // items_per_page)
        if total_pages > 1:
            page = st.slider("Page:", 1, total_pages, 1) - 1
        else:
            page = 0

        start_idx = page * items_per_page
        end_idx = start_idx + items_per_page
        page_particles = all_particles[start_idx:end_idx]

        # Gallery grid (6 columns for performance)
        cols = st.columns(6)
        for i, match in enumerate(page_particles):
            with cols[i % 6]:
                img_name = match["img"]
                pidx = match["idx"]
                p = match["particle"]
                key = match["key"]

                f = st.session_state.uploaded_files_cache.get(img_name)
                if f:
                    img = Image.open(f)
                    img_np = np.array(img)

                    # Crop with tight margin
                    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                    margin = 15
                    crop_x1 = max(0, x - margin)
                    crop_y1 = max(0, y - margin)
                    crop_x2 = min(img_np.shape[1], x + w + margin)
                    crop_y2 = min(img_np.shape[0], y + h + margin)
                    crop = img_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                    # Draw mask bounds if available
                    try:
                        if p.get("mask_x_min") is not None and p.get("mask_x_max") is not None:
                            from PIL import ImageDraw, Image as PILImage
                            crop_pil = Image.fromarray(crop.astype(np.uint8)).convert('RGB')

                            # Convert mask bounds from full image to crop local coords
                            mx1 = int(max(0, p["mask_x_min"] - crop_x1))
                            my1 = int(max(0, p["mask_y_min"] - crop_y1))
                            mx2 = int(min(crop.shape[1], p["mask_x_max"] - crop_x1 + 1))
                            my2 = int(min(crop.shape[0], p["mask_y_max"] - crop_y1 + 1))

                            # Draw green outline (try simpler approach)
                            if mx1 < mx2 and my1 < my2 and mx2 - mx1 > 1 and my2 - my1 > 1:
                                draw = ImageDraw.Draw(crop_pil)
                                # Draw thick outline
                                for offset in range(3):
                                    draw.rectangle(
                                        [(mx1+offset, my1+offset), (mx2-offset, my2-offset)],
                                        outline=(0, 255, 0)
                                    )
                                crop = np.array(crop_pil)
                                if crop.ndim == 3 and crop.shape[2] == 3:
                                    crop = crop[:, :, :3]  # Ensure RGB
                    except Exception as e:
                        pass

                    # Display crop with mask bounds
                    st.image(crop, use_column_width=True, caption=f"{p['diameter_um']}µm")

                    # Info (show sizing method AND mask bounds for debugging)
                    method = p.get("size_method", "?")
                    mask_x_min = p.get("mask_x_min")
                    mask_x_max = p.get("mask_x_max")
                    debug_info = f"{p['class']} | {p['size_bin']}\n({method})"
                    if mask_x_min is not None:
                        debug_info += f"\n🟢 {int(mask_x_min)},{int(mask_x_max)}"
                    st.caption(debug_info)

                    # Inline select checkbox
                    is_selected = key in st.session_state.selected_particles
                    if st.checkbox("Select", value=is_selected, key=f"sel_{key}"):
                        st.session_state.selected_particles.add(key)
                    else:
                        st.session_state.selected_particles.discard(key)

                    # Change class
                    new_cls = st.selectbox(
                        "Class:",
                        ["Fiber", "Glass", "Metallic", "Other"],
                        index=["Fiber", "Glass", "Metallic", "Other"].index(p["class"]),
                        key=f"cls_{key}"
                    )
                    if new_cls != p["class"] and st.button("✓", key=f"save_{key}"):
                        push_undo()
                        st.session_state.results[img_name][pidx]["class"] = new_cls
                        st.session_state.results[img_name][pidx]["size_bin"] = get_size_bin(p["diameter_um"])
                        st.rerun()

                    # Delete
                    if st.button("🗑️ Delete", key=f"del_{key}"):
                        push_undo()
                        st.session_state.results[img_name][pidx]["deleted"] = True
                        st.rerun()

                    if p["black_bg"]:
                        st.warning("⚫ Black BG", icon="⚫")

                    # View full image with zoom/pan (on demand)
                    if st.button("🔍 View Full", key=f"view_{key}"):
                        st.session_state[f"show_full_{key}"] = True

        # Full image viewer (only renders if clicked)
        for match in page_particles:
            key = match["key"]
            if st.session_state.get(f"show_full_{key}", False):
                img_name = match["img"]
                pidx = match["idx"]
                p = match["particle"]

                with st.expander(f"Full Image: {img_name}", expanded=True):
                    f = st.session_state.uploaded_files_cache.get(img_name)
                    if f:
                        img = Image.open(f)
                        img_np = np.array(img)

                        # Create Plotly figure with zoom/pan
                        fig = go.Figure()
                        fig.add_trace(go.Image(z=img_np, name="Image"))

                        # Highlight particle
                        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                        fig.add_shape(
                            type="rect",
                            x0=x, y0=y, x1=x + w, y1=y + h,
                            line=dict(color="lime", width=3)
                        )

                        fig.update_layout(
                            title=f"{img_name} | {p['class']} ({p['diameter_um']}µm) [{p.get('size_method', '?')}]",
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=0, l=0, r=0, t=40),
                            height=600,
                        )
                        fig.update_xaxes(scaleanchor="y", scaleratio=1)
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)

                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Class:** {p['class']}")
                        with col2:
                            st.write(f"**Size:** {p['diameter_um']}µm ({p['size_bin']})")
                        with col3:
                            if st.button("Close", key=f"close_{key}"):
                                st.session_state[f"show_full_{key}"] = False
                                st.rerun()

        st.divider()

        # ─────────────────────────────────────────────────────────────────────
        # PART 3: MASS EDIT
        # ─────────────────────────────────────────────────────────────────────

        if st.session_state.selected_particles:
            st.subheader("⚙️ Bulk Edit Selected")

            selected_count = len(st.session_state.selected_particles)
            st.info(f"**{selected_count} particle(s) selected**")

            col1, col2 = st.columns(2)
            with col1:
                action = st.radio("Action:", ["Delete All Selected", "Change Class To"], horizontal=True)

            with col2:
                if action == "Change Class To":
                    new_cls = st.selectbox("New class:", ["Fiber", "Glass", "Metallic", "Other"], key="mass_cls")

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
        st.info("No particles match filters")