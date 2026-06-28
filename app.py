import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D

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
    "ResNet50V2": ("az_model_resnet50v2.h5", H5_RESNET50V2_ID),
    "ResNet101V2": ("az_model_resnet101v2.h5", H5_RESNET101V2_ID),
    "ResNet152V2": ("az_model_resnet152v2.h5", H5_RESNET152V2_ID),
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
        raise FileNotFoundError(f"File missing after download: {path}")
    size = os.path.getsize(path)
    if size < 50_000 or _is_probably_html(path):
        raise RuntimeError(
            "Downloaded file is not a valid .h5.\n\n"
            "Fix:\n"
            "1) Google Drive -> Share -> Anyone with the link (Viewer)\n"
            "2) Verify FILE IDs are correct"
        )

def _download_if_needed(file_id: str, filename: str) -> str:
    if not file_id or "PASTE_" in file_id:
        raise RuntimeError("Model IDs are not set. Check Google Drive FILE IDs.")

    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    if os.path.exists(local_path):
        try:
            _validate_download(local_path)
            return local_path
        except Exception:
            try:
                os.remove(local_path)
            except Exception:
                pass

    url = gdrive_direct_url(file_id)
    gdown.download(url, local_path, quiet=True, fuzzy=True)
    _validate_download(local_path)
    return local_path

# ============================================================
# 1. BUILD MODELS
# ============================================================
def _build_head(backbone):
    return tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

def build_model_by_name(model_name: str):
    if model_name == "ResNet50V2":
        base = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)
    if model_name == "ResNet101V2":
        base = tf.keras.applications.ResNet101V2(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)
    if model_name == "ResNet152V2":
        base = tf.keras.applications.ResNet152V2(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)
    raise ValueError(f"Unknown model_name: {model_name}")

def load_model_weights_safe(model_name: str, h5_path: str):
    model = build_model_by_name(model_name)
    _ = model(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False)
    model.load_weights(h5_path)
    return model

@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    fname, file_id = MODEL_FILES[model_name]
    local_path = _download_if_needed(file_id, fname)
    return load_model_weights_safe(model_name, local_path)

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
# 3. GRAD-CAM HELPERS
# ============================================================
def get_last_conv_layer_name(model):
    conv_types = (Conv2D, DepthwiseConv2D, SeparableConv2D)

    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, conv_types):
                    return sub.name

    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            continue
        if len(out_shape) == 4:
            return layer.name

    raise ValueError("No suitable conv layer found in model.")

def make_gradcam_heatmap(model, img_array, class_index=None):
    last_conv_name = get_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = tf.convert_to_tensor(preds)

        if class_index is None:
            class_index = tf.argmax(preds[0])

        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(heatmap, pil_image, alpha=0.4):
    import cv2
    img = np.array(pil_image).astype("float32") / 255.0
    h, w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = heatmap_color.astype("float32") / 255.0

    overlay = alpha * heatmap_color + (1 - alpha) * img
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay

# ============================================================
# 4. ENSEMBLE HELPERS (Auto-combine all 3 models equally)
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
# 5. MODERN UI STYLING
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
# 6. SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚙️ Controls")

mode = st.sidebar.radio(
    "Select mode",
    ["🔬 Single Model", "🧠 Ensemble All 3"],
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
# 7. HERO HEADER
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>🧠 FCI-ResNetV2 Alzheimer MRI</h1>
  <p>
    Advanced CNN ensemble with <b>ResNet50V2, ResNet101V2, ResNet152V2</b> and 
    <b>Grad-CAM visualization</b> for clinical brain MRI analysis. 
    Models auto-download from Google Drive.
  </p>
  <div class="pills">
    <span class="pill">ResNet50V2</span>
    <span class="pill">ResNet101V2</span>
    <span class="pill">ResNet152V2</span>
    <span class="pill">Ensemble Mode</span>
    <span class="pill">Grad-CAM</span>
    <span class="pill">Google Drive</span>
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
    1) Pick <b>Ensemble All 3</b> (recommended) or a single ResNetV2 model<br/>
    2) Upload an MRI or select from gallery<br/>
    3) Click <b>▶ Run Prediction</b> to see probabilities + Grad-CAM heatmap<br/><br/>
    <b style="color: #667eea;">💡 Tip:</b> Ensemble mode combines all 3 models for better accuracy!
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
# 8. INFERENCE + GRADCAM
# ============================================================
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

try:
    with st.spinner("🔄 Running prediction…"):
        if mode == "🧠 Ensemble All 3":
            models_dict = load_all_models()
            pred_idx, probs, chosen_key, grad_model = ensemble_predict_simple(models_dict, batch)
            cam_title = "Ensemble All 3"
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

    with st.spinner("🎨 Computing Grad-CAM…"):
        heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
        overlay = overlay_heatmap_on_image(heatmap, orig_img, alpha=0.45)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# ============================================================
# 9. OUTPUT SECTION
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
axes[0].imshow(orig_img)
axes[0].set_title(f"Original MRI\n{pred_class}", fontsize=12, fontweight="bold", color="#667eea")
axes[0].axis("off")

# Grad-CAM overlay
axes[1].imshow(overlay)
axes[1].set_title(f"Grad-CAM Heatmap\n{cam_title}", fontsize=12, fontweight="bold", color="#667eea")
axes[1].axis("off")

# Probabilities bar chart
colors = ["#667eea" if i == pred_idx else "#cbd5e1" for i in range(len(CLASS_NAMES))]
axes[2].barh(CLASS_NAMES, probs, color=colors, edgecolor="white", linewidth=1.5)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Probability", fontsize=11, fontweight="bold")
axes[2].set_title("Class Probabilities", fontsize=12, fontweight="bold", color="#667eea")
axes[2].grid(axis="x", alpha=0.2, linestyle="--")

for i, cls in enumerate(CLASS_NAMES):
    axes[2].text(
        probs[i] + 0.018,
        i,
        f"{probs[i]:.1%}",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#667eea" if i == pred_idx else "#0b1220"
    )

plt.tight_layout(pad=2.0)
st.pyplot(fig)

# Download button
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=180, facecolor="white")
buf.seek(0)

st.download_button(
    "💾 Save result image",
    data=buf,
    file_name=f"result_{pred_class}.png",
    mime="image/png"
)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eef0f6; color: #5b6475; font-size: 0.9rem;">
      <b>FCI-ResNetV2 Alzheimer Detection System</b> | Clinical MRI Analysis | Powered by TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
