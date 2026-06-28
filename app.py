import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown
from scipy.ndimage import gaussian_filter
import cv2

# ============================================================
# 0. BASIC SETUP
# ============================================================
st.set_page_config(
    page_title="FCI-ResNetV2 Alzheimer MRI",
    layout="wide"
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Google Drive IDs (.h5)
H5_RESNET50V2_ID = "1H7iUcI94ul01BWqtGwf-LZ2jKnATf17l"
H5_RESNET101V2_ID = "11UB7-lHDqcAuPNnA_q8wAQrgS2CijHV6"
H5_RESNET152V2_ID = "1Rg0d2efE-oV_usoJkr6F2xXLCRW9yPTi"

MODEL_FILES = {
    "ResNet50V2": ("resnet50v2.keras", H5_RESNET50V2_ID),
    "ResNet101V2": ("resnet101v2.keras", H5_RESNET101V2_ID),
    "ResNet152V2": ("resnet152v2.keras", H5_RESNET152V2_ID),
}

def gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"

def _is_probably_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4096).lower()
        return (b"<html" in head) or (b"<!doctype html" in head) or (b"google drive" in head)
    except Exception:
        return False

def _validate_download(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    size = os.path.getsize(path)
    if size < 50000:
        raise RuntimeError("Downloaded model looks invalid. Check Google Drive sharing and File ID.")

def _download_if_needed(file_id: str, filename: str) -> str:
    local_path = os.path.join(MODEL_CACHE_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    url = gdrive_direct_url(file_id)
    gdown.download(url=url, output=local_path, quiet=False)
    _validate_download(local_path)
    return local_path

# ============================================================
# 1. LOAD MODELS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_single_model(model_name):
    filename, file_id = MODEL_FILES[model_name]
    local_path = _download_if_needed(file_id, filename)

    model = tf.keras.models.load_model(local_path, compile=False)

    return model

@st.cache_resource(show_spinner=False)
def load_all_models():
    return {
        "ResNet50V2": load_single_model("ResNet50V2"),
        "ResNet101V2": load_single_model("ResNet101V2"),
        "ResNet152V2": load_single_model("ResNet152V2"),
    }

# ============================================================
# 2. IMAGE HELPERS
# ============================================================
def load_image_from_file(file, img_size=IMG_SIZE):
    if isinstance(file, str):
        img = Image.open(file).convert("RGB")
    else:
        img = Image.open(file).convert("RGB")
    img_resized = img.resize(img_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return img, batch

# ============================================================
# 3. BRAIN TISSUE MASKING (FROM NOTEBOOK)
# ============================================================
def tissue_mask_from_rgb(rgb01: np.ndarray) -> np.ndarray:
    """Extract brain tissue mask from RGB image."""
    gray = rgb01.mean(axis=2)
    vals = gray[gray > 0]
    if vals.size == 0:
        return np.ones_like(gray, dtype=np.float32)
    thr = max(0.02, np.percentile(vals, 5))
    m = (gray > thr).astype(np.float32)
    m = gaussian_filter(m, sigma=1.5)
    return np.clip(m, 0, 1)

def robust_norm(cam, mask=None, p_low=2, p_high=98):
    """Percentile-based normalization."""
    cam = cam.astype(np.float32)
    if mask is not None:
        vals = cam[mask > 0.2]
        if vals.size < 50:
            vals = cam.reshape(-1)
    else:
        vals = cam.reshape(-1)
    lo, hi = np.percentile(vals, [p_low, p_high])
    out = (cam - lo) / (hi - lo + 1e-8)
    out = np.clip(out, 0, 1)
    if mask is not None:
        out *= mask
        if out.max() > 0:
            out = out / (out.max() + 1e-8)
    return out

def process_cam_focus(cam01, out_h, out_w, tissue_mask01, sigma=2.0, thr_pct=80, power=0.9):
    """Process CAM with tissue masking, smoothing, thresholding."""
    cam = np.clip(cam01.astype(np.float32), 0, 1)
    
    # Resize to output dimensions
    cam = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)).astype(np.float32) / 255.0
    
    # Smooth
    cam = gaussian_filter(cam, sigma=float(sigma))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    
    # Apply tissue mask
    cam = cam * np.clip(tissue_mask01, 0, 1)
    cam = robust_norm(cam, tissue_mask01, p_low=2, p_high=98)
    
    # Threshold
    vals = cam[cam > 0]
    if vals.size > 20:
        thr = np.percentile(vals, float(thr_pct))
        cam = np.where(cam >= thr, cam, 0.0)
    
    # Normalize
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    
    # Final smooth
    cam = gaussian_filter(cam, sigma=max(1.0, float(sigma)*0.6))
    if cam.max() > 0:
        cam = cam / (cam.max() + 1e-8)
    
    # Power adjustment for contrast
    cam = cam ** float(power)
    return cam

# ============================================================
# 4. ADVANCED GRAD-CAM (FROM NOTEBOOK)
# ============================================================
def find_last_conv2d_layer_name(model):
    """Find the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("No Conv2D-like layer found for this model.")

def gradcam_heatmap(model, img_tensor, class_index, target_layer_name):

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(target_layer_name).output,
            model.outputs[0],
        ],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor, training=False)

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.nn.relu(heatmap)

    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap
    
def make_gradcam_heatmap_with_focus(model, img_array, orig_img_rgb01, class_index=None, 
                                     sigma=2.0, thr_pct=80, power=0.9):
    """Compute Grad-CAM with tissue masking and processing."""
    target_layer = find_last_conv2d_layer_name(model)
    cam_small = gradcam_heatmap(model, img_array, class_index, target_layer).numpy()
    
    H, W = orig_img_rgb01.shape[:2]
    tissue_mask = tissue_mask_from_rgb(orig_img_rgb01)
    
    cam_focus = process_cam_focus(
        cam_small, out_h=H, out_w=W, tissue_mask01=tissue_mask,
        sigma=sigma, thr_pct=thr_pct, power=power
    )
    return cam_focus

def overlay_with_alpha(base_rgb01, cam01, alpha=0.50, cmap="jet"):
    """Overlay heatmap on image with alpha blending."""
    cmap_fn = plt.get_cmap(cmap)
    cam_rgb = cmap_fn(np.clip(cam01, 0, 1))[:, :, :3]
    alpha_map = alpha * (cam01 ** 0.7)
    alpha_map = alpha_map * (cam01 > 0).astype(np.float32)
    overlay = base_rgb01 * (1 - alpha_map[..., None]) + cam_rgb * alpha_map[..., None]
    return np.clip(overlay, 0, 1)

# ============================================================
# 5. ENSEMBLE HELPERS
# ============================================================
def ensemble_predict_simple(models_dict, img_batch):
    """Simple averaging ensemble of all 3 models."""
    keys = ["ResNet50V2", "ResNet101V2", "ResNet152V2"]
    preds_list = []
    
    for k in keys:
        p = models_dict[k].predict(img_batch, verbose=0)[0]
        preds_list.append(p)
    
    # Average predictions
    avg_preds = np.mean(preds_list, axis=0).astype(np.float32)
    pred_idx = int(np.argmax(avg_preds))
    
    # Pick model with highest confidence for Grad-CAM
    best_idx = 0
    best_conf = preds_list[0][pred_idx]
    for i in range(1, 3):
        if preds_list[i][pred_idx] > best_conf:
            best_conf = preds_list[i][pred_idx]
            best_idx = i
    
    return pred_idx, avg_preds, keys[best_idx], models_dict[keys[best_idx]]

# ============================================================
# 6. MODERN UI STYLING
# ============================================================
st.markdown(
    """
<style>
:root{
  --bg: #ffffff;
  --ink: #0b1220;
  --muted: #5b6475;
  --line: #eef0f6;
  --card: rgba(255,255,255,0.86);
  --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
  --shadow2: 0 10px 24px rgba(15, 23, 42, 0.06);
  --radius: 20px;
}

html, body, [class*="css"] { 
  font-family: 'Segoe UI', Trebuchet MS, sans-serif !important;
}
section.main { background: var(--bg); }
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1220px; }

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(135deg, #fbfbff 0%, #f3f4f9 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--ink);
  font-weight: 950;
}
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] label {
  color: #111827 !important;
  font-weight: 700;
  font-size: 0.95rem;
}
[data-testid="stSidebar"] .stButton button{
  width: 100%;
  border-radius: 14px;
  padding: 0.85rem 1rem;
  font-weight: 950;
  border: none;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  box-shadow: var(--shadow2);
  font-size: 1.05rem;
}
[data-testid="stSidebar"] .stButton button:hover{ 
  filter: brightness(1.1);
  transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(102, 126, 234, 0.3);
}

/* Hero */
.hero{
  position: relative;
  border: 2px solid var(--line);
  border-radius: 28px;
  padding: 2.5rem 2rem;
  box-shadow: var(--shadow);
  overflow: hidden;
  background: radial-gradient(1200px 400px at 15% 0%, rgba(102, 126, 234, 0.22), transparent 60%),
              radial-gradient(1200px 400px at 85% 100%, rgba(118, 75, 162, 0.16), transparent 60%),
              linear-gradient(180deg, #ffffff 0%, #fbfbff 100%);
}
.hero h1{
  margin: 0;
  font-size: 2.8rem;
  letter-spacing: -0.05em;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 950;
}
.hero p{
  margin: 0.6rem 0 0;
  color: var(--muted);
  font-size: 1.08rem;
  line-height: 1.65;
  max-width: 980px;
}
.pills{ 
  margin-top: 1.2rem; 
  display: flex; 
  flex-wrap: wrap; 
  gap: 0.6rem; 
}
.pill{
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  font-size: 0.88rem;
  border: 1.5px solid var(--line);
  background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,254,0.8) 100%);
  color: #667eea;
  font-weight: 800;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
}

/* Cards */
.card{
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: var(--radius);
  padding: 1.4rem;
  box-shadow: var(--shadow2);
  backdrop-filter: blur(10px);
}
.card-title{
  font-size: 1.3rem;
  font-weight: 950;
  margin: 0 0 0.8rem;
  color: var(--ink);
}
.small{
  color: var(--muted);
  font-size: 0.98rem;
  line-height: 1.6;
}

/* Metric chips */
.chips{ 
  display: flex; 
  flex-wrap: wrap; 
  gap: 0.6rem; 
  margin-top: 0.4rem; 
}
.chip{
  border: 1.5px solid #e9edf7;
  border-radius: 999px;
  padding: 0.5rem 0.9rem;
  background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(248,250,254,0.6) 100%);
  font-weight: 900;
  color: #667eea;
  font-size: 0.9rem;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
}

/* Buttons */
div.stDownloadButton button{
  border-radius: 14px !important;
  padding: 0.8rem 1.2rem !important;
  font-weight: 950 !important;
  border: none !important;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  color: white !important;
  box-shadow: var(--shadow2);
}
div.stDownloadButton button:hover{ 
  filter: brightness(1.1) !important;
  transform: translateY(-2px);
}

footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# 7. SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚙️ Controls")

mode = st.sidebar.radio(
    "Select mode",
    ["🔬 Single Model", "🧠 FCI Ensemble"],
    index=1
)

source = st.sidebar.radio(
    "Image source",
    ["📤 Upload MRI", "🖼️ Sample gallery"],
    index=1
)

chosen_file = None

if source == "📤 Upload MRI":
    uploaded = st.sidebar.file_uploader(
        "Upload an MRI image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        chosen_file = uploaded
else:
    SAMPLE_DIR = "sample_images"
    try:
        files = sorted(
            [f for f in os.listdir(SAMPLE_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
    except FileNotFoundError:
        files = []

    if files:
        sample_name = st.sidebar.selectbox("Pick a sample", files)
        chosen_file = os.path.join(SAMPLE_DIR, sample_name)
    else:
        st.sidebar.warning("No images in sample_images/")

if mode == "🔬 Single Model":
    st.sidebar.markdown("---")
    selected_model = st.sidebar.selectbox(
        "Choose model",
        ["ResNet50V2", "ResNet101V2", "ResNet152V2"],
        index=0
    )
else:
    selected_model = None

run_button = st.sidebar.button("▶ Run Prediction", use_container_width=True)

# ============================================================
# 8. HERO HEADER
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>🧠 FCI-ResNetV2 Alzheimer MRI</h1>
  <p>
    Advanced CNN ensemble with <b>ResNet50V2, ResNet101V2, ResNet152V2</b> and 
    <b>Grad-CAM+ </b> for clinical MRI analysis. 
  </p>
  <div class="pills">
    <span class="pill">ResNet50V2</span>
    <span class="pill">ResNet101V2</span>
    <span class="pill">ResNet152V2</span>
    <span class="pill">FCI Ensemble</span>
    <span class="pill">Grad-CAM+</span>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# Pre-run info card
# ============================================================
if not run_button:
    st.markdown(
        """
<div class="card" style="margin-top: 1.5rem;">
  <div class="card-title">✨ How to Use</div>
  <div class="small">
    1) Pick <b>FCI Ensemble</b> (recommended) or a single ResNetV2 model<br/>
    2) Upload an MRI or select from gallery<br/>
    3) Click <b>▶ Run Prediction</b> to see probabilities + advanced Grad-CAM heatmap<br/><br/>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
    st.stop()

if chosen_file is None:
    st.error("❌ Please upload or select an image first.")
    st.stop()

# ============================================================
# 9. INFERENCE + ADVANCED GRADCAM
# ============================================================
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)
orig_img_arr = np.array(orig_img).astype("float32") / 255.0

try:
    with st.spinner("🔄 Running prediction…"):
        if mode == "🧠 FCI Ensemble":
            models_dict = load_all_models()
            pred_idx, probs, chosen_key, grad_model = ensemble_predict_simple(models_dict, batch)
            cam_title = "FCI Ensemble"
            mode_label = "Ensemble Mode"
        else:
            grad_model = load_single_model(selected_model)
            probs = grad_model.predict(batch, verbose=0)[0].astype(np.float32)
            pred_idx = int(np.argmax(probs))
            chosen_key = selected_model
            cam_title = selected_model
            mode_label = "Single Model"

    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(probs))

    with st.spinner("🎨 Computing advanced Grad-CAM…"):
        heatmap = make_gradcam_heatmap_with_focus(
            grad_model, batch, orig_img_arr, 
            class_index=pred_idx,
            sigma=2.0, thr_pct=80, power=0.9
        )
        overlay = overlay_with_alpha(orig_img_arr, heatmap, alpha=0.45, cmap="jet")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# ============================================================
# 10. OUTPUT SECTION
# ============================================================
st.markdown(
    f"""
    <div class="card" style="margin-top: 1.5rem;">
      <div class="card-title">📊 Prediction Output</div>
      <div class="chips">
        <span class="chip"><b>🔬 Mode:</b> {mode_label}</span>
        <span class="chip"><b>📄 Model:</b> {cam_title}</span>
        <span class="chip"><b>✅ Prediction:</b> {pred_class}</span>
        <span class="chip"><b>📈 Confidence:</b> {confidence:.1%}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# VISUALIZATION
# ============================================================
st.markdown('<div class="card" style="margin-top: 1.3rem;">', unsafe_allow_html=True)

fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 4.8),
    gridspec_kw={"width_ratios": [1, 1, 0.95]},
    facecolor="white"
)

# Original
axes[0].imshow(orig_img_arr)
axes[0].set_title(f"Original MRI\n{pred_class}", fontsize=12, fontweight="bold", color="#667eea")
axes[0].axis("off")

# Grad-CAM heatmap
axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
axes[1].set_title(f"Grad-CAM+ Heatmap\n{cam_title}", fontsize=12, fontweight="bold", color="#667eea")
axes[1].axis("off")

# Overlay with contours
axes[2].imshow(overlay)
axes[2].contour(heatmap, levels=[0.35, 0.55, 0.75], linewidths=1.0, colors="white", alpha=0.8)
axes[2].set_title("Overlay + Contours", fontsize=12, fontweight="bold", color="#667eea")
axes[2].axis("off")

plt.tight_layout(pad=2.0)
st.pyplot(fig)

# Probabilities bar chart
fig2, ax = plt.subplots(figsize=(10, 4), facecolor="white")
colors = ["#667eea" if i == pred_idx else "#cbd5e1" for i in range(len(CLASS_NAMES))]
ax.barh(CLASS_NAMES, probs, color=colors, edgecolor="white", linewidth=1.5)
ax.set_xlim(0, 1)
ax.set_xlabel("Probability", fontsize=11, fontweight="bold")
ax.set_title("Class Probabilities", fontsize=12, fontweight="bold", color="#667eea")
ax.grid(axis="x", alpha=0.2, linestyle="--")

for i, cls in enumerate(CLASS_NAMES):
    ax.text(
        probs[i] + 0.018,
        i,
        f"{probs[i]:.1%}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#667eea" if i == pred_idx else "#0b1220"
    )

plt.tight_layout()
st.pyplot(fig2)

# Download buttons
buf1 = io.BytesIO()
fig.savefig(buf1, format="png", bbox_inches="tight", dpi=180, facecolor="white")
buf1.seek(0)

buf2 = io.BytesIO()
fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=180, facecolor="white")
buf2.seek(0)

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "💾 Save Grad-CAM visualization",
        data=buf1,
        file_name=f"gradcam_{pred_class}.png",
        mime="image/png",
        on_click="ignore"
    )
with col2:
    st.download_button(
        "💾 Save probability chart",
        data=buf2,
        file_name=f"probabilities_{pred_class}.png",
        mime="image/png",
        on_click="ignore"
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eef0f6; color: #5b6475; font-size: 0.9rem;">
      <b>FCI-ResNetV2 Alzheimer Detection System</b> | Advanced Grad-CAM+ | Powered by TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
