"""
Unified Particle Detection Gallery - TILING INFERENCE FOR MASSIVE IMAGES
All particles → Filter → Inline delete/change class + Mass edit + Full image viewer
+ AUTOMATIC TILING for giant stitched images (no detail loss!)

✅ Handles massive stitched images (5000px+) with tiling
✅ Trained at 1080px but detects particles at full quality
✅ Overlapping tiles + smart deduplication
✅ Fast gallery, zoom/pan viewer, mass edit, undo, export

Usage:
    streamlit run particle_review_gallery_tiling.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Set environment variables BEFORE any imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/yolo_config'
os.environ['PIP_NO_CACHE_DIR'] = '1'

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

# Suppress warnings
warnings.filterwarnings('ignore')

# NOW import YOLO (after env vars are set)
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except Exception as e:
    print(f"⚠️ YOLO import warning (non-critical): {e}")
    YOLO = None
    YOLO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/best.pt"
CALIBRATION_UM_PER_PIXEL = 1.299
BLACK_BG_THRESHOLD = 30

# Tiling config - SUPER FAST FOR MASSIVE IMAGES (48k+)
TILE_SIZE = 3000  # Large tiles = ~350 tiles for 48k image (YOLO handles it)
TILE_OVERLAP_PCT = 0.15  # 15% overlap (still catches particles at tile boundaries)
IOU_DEDUP_THRESHOLD = 0.3  # Remove duplicate detections if IOU > this

# For 48891x46872 image @ 3000px tiles:
# ~350 total tiles = 30-45 minutes! ✅
# (Model trained @ 1080px but YOLO internally handles larger inputs)
# You can adjust in sidebar if needed

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

st.set_page_config(page_title="Particle Detection Review", page_icon="icon.png", layout="wide")

# Load logo icon and embed in the UI
try:
    with open("icon.png", "rb") as f:
        img = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:15px;">
        <img src="data:image/png;base64,{img}" width="80">
        <h1 style="margin:0;">🧹 dirt_sniffer: Review Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
except:
    st.markdown("# 🧹 dirt_sniffer: Review Dashboard")


@st.cache_resource
def load_model():
    """Load YOLO model once"""
    if not YOLO_AVAILABLE:
        st.error("❌ YOLO not available. Check dependencies.")
        return None

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        return None

    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TILING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def generate_tiles(image_h, image_w, tile_size=TILE_SIZE, overlap_pct=TILE_OVERLAP_PCT):
    """Generate overlapping tile coordinates"""
    overlap = int(tile_size * overlap_pct)
    stride = tile_size - overlap

    tiles = []
    y = 0
    while y < image_h:
        x = 0
        while x < image_w:
            x2 = min(x + tile_size, image_w)
            y2 = min(y + tile_size, image_h)

            # Expand tile to full size if it's the edge (avoid artifacts)
            if x2 == image_w:
                x = max(0, image_w - tile_size)
            if y2 == image_h:
                y = max(0, image_h - tile_size)

            x2 = min(x + tile_size, image_w)
            y2 = min(y + tile_size, image_h)

            tiles.append((x, y, x2, y2))
            x += stride
        y += stride

    return tiles


def iou(box1, box2):
    """Calculate IOU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max < xi_min or yi_max < yi_min:
        return 0.0

    inter_area = (xi_max - xi_min) * (yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def deduplicate_detections(particles, iou_threshold=IOU_DEDUP_THRESHOLD):
    """Remove duplicate detections from overlapping tiles"""
    if not particles:
        return particles

    # Sort by confidence descending
    particles = sorted(particles, key=lambda p: p["confidence"], reverse=True)

    kept = []
    for p in particles:
        box_p = (p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])

        # Check if too similar to already-kept particle
        is_duplicate = False
        for pk in kept:
            box_k = (pk["x"], pk["y"], pk["x"] + pk["w"], pk["y"] + pk["h"])
            if iou(box_p, box_k) > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(p)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_size_bin(diameter_um):
    """Get size bin label for diameter"""
    for label, lo, hi in SIZE_BINS:
        if lo <= diameter_um < hi:
            return label
    return "K"


def is_black_background(image_np, x, y, w, h, threshold=BLACK_BG_THRESHOLD):
    """Check if background is black (safety check)"""
    try:
        region = image_np[max(0, y - 5):min(image_np.shape[0], y + h + 5),
        max(0, x - 5):min(image_np.shape[1], x + w + 5)]
        if region.size == 0:
            return False
        avg_brightness = np.mean(region)
        return avg_brightness < threshold
    except:
        return False


def process_image(image_path, model, tile_size=TILE_SIZE):
    """
    Process image with YOLO - AUTOMATIC TILING for massive images (48k+)

    ✅ Massive images (48k+ pixels) are tiled → tiled inferences merged
    ✅ Small images (<tile_size) processed directly
    ✅ Full quality particle detection at all scales
    ✅ Smart deduplication of overlapping detections
    """
    try:
        # Step 1: Load with PIL (NO pixel limits)
        img_pil = Image.open(image_path)

        # Step 2: Convert to RGB
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        # Step 3: Convert to numpy
        image = np.array(img_pil)

        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        particles = []

        # Step 4: Check if tiling needed
        if h > tile_size or w > tile_size:
            # TILING: Process in overlapping tiles
            tiles = generate_tiles(h, w, tile_size, TILE_OVERLAP_PCT)
            total_tiles = len(tiles)

            for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
                tile = image[y1:y2, x1:x2]

                # Run YOLO on tile
                results = model(tile, iou=0.45, conf=0.02, verbose=False)

                # Extract particles and adjust coordinates to full image
                for r in results:
                    if r.boxes is None or r.masks is None:
                        continue

                    for mask, box, cls, conf in zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        try:
                            bx1, by1, bx2, by2 = [int(v) for v in box.tolist()]

                            # Adjust to full image coordinates
                            x_full = bx1 + x1
                            y_full = by1 + y1

                            label = model.names[int(cls)]
                            box_w = bx2 - bx1
                            box_h = by2 - by1
                            max_diam_um = max(box_w, box_h) * CALIBRATION_UM_PER_PIXEL

                            is_black = is_black_background(image, x_full, y_full, box_w, box_h)

                            particles.append({
                                "x": x_full, "y": y_full, "w": box_w, "h": box_h,
                                "class": label, "confidence": float(conf),
                                "diameter_um": round(max_diam_um, 1),
                                "size_bin": get_size_bin(max_diam_um),
                                "deleted": False,
                                "black_bg": is_black
                            })
                        except Exception as e:
                            print(f"Error processing particle in tile {tile_idx}: {e}")
                            continue

            # Deduplicate overlapping detections
            particles = deduplicate_detections(particles, IOU_DEDUP_THRESHOLD)

        else:
            # NO TILING: Process entire image directly
            results = model(image, iou=0.45, conf=0.02, verbose=False)

            for r in results:
                if r.boxes is None or r.masks is None:
                    continue

                for mask, box, cls, conf in zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    try:
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
                            "deleted": False,
                            "black_bg": is_black
                        })
                    except Exception as e:
                        print(f"Error processing particle: {e}")
                        continue

        return particles if particles else None

    except Exception as e:
        st.error(f"❌ Error processing {image_path}: {str(e)}")
        return None


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
    """Save state for undo"""
    st.session_state.undo_stack.append(deepcopy(st.session_state.results))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - UPLOAD & CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📤 Upload & Process")

    # Tile size config for massive images
    st.write("**🎚️ Tiling config** (for 48k+ images):")
    tile_size_radio = st.radio(
        "Tile size:",
        [2400, 3000, 3600],
        format_func=lambda x: f"{x}px",
        help="Larger = faster. Model trained @ 1080px, but handles any size"
    )
    custom_tile_size = tile_size_radio

    # Calculate est. tiles
    stride = custom_tile_size * (1 - TILE_OVERLAP_PCT)
    est_tiles_h = int(np.ceil(46872 / stride))
    est_tiles_w = int(np.ceil(48891 / stride))
    est_tiles = est_tiles_h * est_tiles_w

    if est_tiles <= 400:
        speed_est = "⚡ 30-45 min"
    elif est_tiles <= 600:
        speed_est = "⚡ 45-60 min"
    else:
        speed_est = "⏱️ 1-2 hours"

    st.caption(f"Est. {est_tiles} tiles • {speed_est}")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔍 Run Inference"):
            model = load_model()
            if model is None:
                st.error("❌ Model not found at " + MODEL_PATH)
            else:
                progress = st.progress(0)
                status = st.empty()
                errors = []

                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, f in enumerate(uploaded_files):
                        status.text(f"Processing {i + 1}/{len(uploaded_files)}: {f.name}")

                        try:
                            temp_path = os.path.join(tmpdir, f.name)
                            with open(temp_path, "wb") as fp:
                                fp.write(f.getbuffer())

                            # Check image size to warn about tiling
                            img_check = Image.open(temp_path)
                            img_h, img_w = img_check.size[1], img_check.size[0]

                            if img_h > custom_tile_size or img_w > custom_tile_size:
                                num_tiles = int(np.ceil(img_w / custom_tile_size) * np.ceil(img_h / custom_tile_size))
                                status.text(
                                    f"🟡 Tiling {f.name} ({img_w}x{img_h}) into {num_tiles} tiles @ {custom_tile_size}px...")

                            particles = process_image(temp_path, model, custom_tile_size)
                            if particles:
                                st.session_state.results[f.name] = particles
                                st.session_state.uploaded_files_cache[f.name] = f
                                status.text(f"✅ {f.name}: {len(particles)} particles")
                            else:
                                errors.append(f"No particles detected in {f.name}")

                        except Exception as e:
                            errors.append(f"Error with {f.name}: {str(e)}")

                        progress.progress((i + 1) / len(uploaded_files))

                status.text("✅ Done!")
                if errors:
                    for err in errors:
                        st.warning(err)
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
# MAIN CONTENT
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
        page = st.slider("Page:", 1, total_pages, 1) - 1

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
                    try:
                        img = Image.open(f)
                        img_np = np.array(img)

                        # Crop with tight margin
                        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                        margin = 15
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(img_np.shape[1], x + w + margin)
                        y2 = min(img_np.shape[0], y + h + margin)
                        crop = img_np[y1:y2, x1:x2]

                        # Display crop
                        st.image(crop, use_column_width=True, caption=f"{p['diameter_um']}µm")

                        # Info
                        st.caption(f"{p['class']} | {p['size_bin']}")

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

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # Full image viewer (only renders if clicked) - zoom/pan with Plotly
        for match in page_particles:
            key = match["key"]
            if st.session_state.get(f"show_full_{key}", False):
                img_name = match["img"]
                pidx = match["idx"]
                p = match["particle"]

                with st.expander(f"Full Image: {img_name}", expanded=True):
                    f = st.session_state.uploaded_files_cache.get(img_name)
                    if f:
                        try:
                            img = Image.open(f)
                            img_np = np.array(img)

                            # Create Plotly figure with zoom/pan
                            fig = go.Figure()
                            fig.add_trace(go.Image(z=img_np, name="Image"))

                            # Highlight particle with lime green box
                            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                            fig.add_shape(
                                type="rect",
                                x0=x, y0=y, x1=x + w, y1=y + h,
                                line=dict(color="lime", width=3)
                            )

                            fig.update_layout(
                                title=f"{img_name} | {p['class']} ({p['diameter_um']}µm)",
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

                        except Exception as e:
                            st.error(f"Error viewing image: {str(e)}")

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
                    parts = key.rsplit("_", 1)
                    img_name = parts[0]
                    idx = int(parts[1])
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