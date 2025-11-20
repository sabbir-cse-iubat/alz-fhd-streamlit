import os
import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D

# ---------------------------------------------------------------------
# 0. BASIC SETUP + HARD UI THEME
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI",
    layout="wide"
)

# 🔹 Strong custom theme (glassmorphism, bold colors)
st.markdown(
    """
    <style>
    /* ===== GLOBAL APP BACKGROUND ===== */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #071633 0, #020617 35%, #020617 100%);
        color: #e5e7eb;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* ===== MAIN CONTAINER WIDTH ===== */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.98);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
        backdrop-filter: blur(18px);
    }
    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f9fafb !important;
        font-weight: 700 !important;
    }

    /* Sidebar radio / select label spacing */
    section[data-testid="stSidebar"] label {
        font-weight: 500 !important;
    }

    /* ===== TITLES ===== */
    h1 {
        font-weight: 800 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 1.6rem !important;
        color: #e5e7eb !important;
    }

    h2, h3, h4 {
        color: #e5e7eb !important;
    }

    /* ===== “CARDS” ===== */
    .hero-card {
        background: radial-gradient(circle at 10% 0%, rgba(45, 212, 191, 0.16) 0, rgba(15, 23, 42, 0.9) 40%, rgba(15, 23, 42, 0.98) 100%);
        border-radius: 1.4rem;
        padding: 1.4rem 1.6rem 1.3rem 1.6rem;
        border: 1px solid rgba(148, 163, 184, 0.45);
        box-shadow: 0 22px 60px rgba(15, 23, 42, 0.9);
    }

    .info-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,64,175,0.9));
        border-radius: 1.1rem;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(129, 140, 248, 0.4);
        box-shadow: 0 18px 40px rgba(15,23,42,0.85);
    }

    .prediction-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.96) 0%, rgba(30,64,175,0.9) 40%, rgba(17,94,163,0.95) 100%);
        border-radius: 1.4rem;
        padding: 1.3rem 1.5rem 1.2rem 1.5rem;
        border: 1px solid rgba(129, 140, 248, 0.55);
        box-shadow: 0 24px 70px rgba(15,23,42,0.95);
    }

    /* Divider line subtle */
    hr {
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.5);
        margin-top: 1.6rem;
        margin-bottom: 1.3rem;
    }

    /* ===== TEXT ACCENTS ===== */
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        color: #f9fafb;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5f5;
        max-width: 40rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(96, 165, 250, 0.6);
        background: rgba(15, 23, 42, 0.65);
        font-size: 0.78rem;
        color: #e5e7eb;
        margin-right: 0.5rem;
    }

    .chip span {
        font-size: 0.8rem;
    }

    .final-pred {
        font-size: 1.1rem;
        font-weight: 650;
        color: #f9fafb;
    }

    /* ===== BUTTONS ===== */
    .stButton>button {
        border-radius: 999px;
        padding: 0.42rem 1.2rem;
        border: 1px solid rgba(59, 130, 246, 0.0);
        font-weight: 600;
        font-size: 0.9rem;
        background: linear-gradient(135deg, #22c55e, #14b8a6);
        color: #020617;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.45);
    }
    .stButton>button:hover {
        box-shadow: 0 16px 35px rgba(34, 211, 238, 0.5);
        transform: translateY(-1px);
    }

    /* Sidebar button variant */
    section[data-testid="stSidebar"] .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        color: #f9fafb;
        border: 1px solid rgba(59,130,246,0.6);
        box-shadow: 0 10px 25px rgba(37,99,235,0.55);
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #22c55e, #0ea5e9);
    }

    /* Download button main area */
    .stDownloadButton>button {
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        font-weight: 600;
        background: radial-gradient(circle at 0 0, #22c55e, #0ea5e9);
        color: #020617;
        padding: 0.45rem 1.3rem;
        box-shadow: 0 14px 32px rgba(34, 197, 94, 0.5);
        font-size: 0.9rem;
    }
    .stDownloadButton>button:hover {
        filter: brightness(1.05);
        box-shadow: 0 18px 40px rgba(6,182,212,0.7);
    }

    /* Tighten matplotlib canvas top margin */
    .stPlotlyChart, .stPyplot {
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_DIR = "sample_images"
os.makedirs(SAMPLE_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. GOOGLE DRIVE MODEL LINKS
# ---------------------------------------------------------------------
DENSENET_ID  = "1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO"
MOBILENET_ID = "1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0"
RESNET_ID    = "1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW"


def gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


# ---------------------------------------------------------------------
# 2. MODEL LOADING VIA GDOWN  (LOGIC UNCHANGED)
# ---------------------------------------------------------------------
def _download_model_if_needed(file_id: str, filename: str) -> str:
    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    if not os.path.exists(local_path):
        url = gdrive_direct_url(file_id)
        try:
            gdown.download(url, local_path, quiet=True)
        except Exception as e:
            st.error(f"Failed to download {filename} from Google Drive.\nError: {e}")
            raise

    if not os.path.exists(local_path):
        st.error(f"Model file {filename} was not created. Check Drive sharing / ID.")
        raise FileNotFoundError(local_path)

    return local_path


@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    if model_name == "DenseNet121":
        file_id = DENSENET_ID
        fname = "densenet_alz_fhd.keras"
    elif model_name == "MobileNetV1":
        file_id = MOBILENET_ID
        fname = "mobilenet_alz_fhd.keras"
    elif model_name == "ResNet50V2":
        file_id = RESNET_ID
        fname = "resnet_alz_fhd.keras"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    local_path = _download_model_if_needed(file_id, fname)

    try:
        model = tf.keras.models.load_model(local_path, compile=False)
    except Exception as e:
        st.error(
            f"Failed to load model from {local_path}.\n"
            f"This usually means the file is not a valid Keras model.\n\nError: {e}"
        )
        raise

    return model


@st.cache_resource(show_spinner=False)
def load_all_base_models():
    dn = load_single_model("DenseNet121")
    mb = load_single_model("MobileNetV1")
    rn = load_single_model("ResNet50V2")
    return {"DenseNet": dn, "MobileNet": mb, "ResNet": rn}


# ---------------------------------------------------------------------
# 3. IMAGE HELPERS (UNCHANGED)
# ---------------------------------------------------------------------
def load_image_from_file(file, img_size=IMG_SIZE):
    if isinstance(file, str):
        img = Image.open(file).convert("RGB")
    else:
        img = Image.open(file).convert("RGB")
    img_resized = img.resize(img_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return img, batch


# ---------------------------------------------------------------------
# 4. GRAD-CAM HELPERS (LOGIC SAME)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 5. FHD ENSEMBLE HELPERS (UNCHANGED)
# ---------------------------------------------------------------------
def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd_single(models_dict, img_batch):
    keys = ["DenseNet", "MobileNet", "ResNet"]
    preds_list = []

    for k in keys:
        preds = models_dict[k].predict(img_batch, verbose=0)[0]
        preds_list.append(preds)

    m_count = len(preds_list)
    avg_fhd = []

    for m1 in range(m_count):
        dists = []
        for m2 in range(m_count):
            if m1 == m2:
                continue
            dists.append(fuzzy_hellinger_distance(preds_list[m1], preds_list[m2]))
        avg_fhd.append(np.mean(dists))

    best_idx = int(np.argmin(avg_fhd))
    chosen_probs = preds_list[best_idx]
    chosen_key = keys[best_idx]
    pred_idx = int(np.argmax(chosen_probs))
    return chosen_probs, pred_idx, chosen_key


def run_fhd_ensemble(img_batch):
    models_dict = load_all_base_models()
    probs, pred_idx, chosen_key = ensemble_predict_fhd_single(models_dict, img_batch)
    grad_model = models_dict[chosen_key]
    return probs, pred_idx, chosen_key, grad_model


# ---------------------------------------------------------------------
# 6. SIDEBAR CONTROLS (LOGIC SAME, JUST COLORS FROM CSS)
# ---------------------------------------------------------------------
st.sidebar.title("⚙ Controls")

model_name = st.sidebar.selectbox(
    "Select model",
    ["DenseNet121", "MobileNetV1", "ResNet50V2", "FHD-HybridNet"],
    index=3
)

source = st.sidebar.radio(
    "Choose image source",
    ["Upload MRI", "Sample gallery"],
    index=1
)

chosen_file = None

if source == "Upload MRI":
    uploaded = st.sidebar.file_uploader(
        "Upload a brain MRI image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        chosen_file = uploaded
else:
    try:
        files = sorted(
            [f for f in os.listdir(SAMPLE_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
    except FileNotFoundError:
        files = []

    if files:
        sample_name = st.sidebar.selectbox("Pick a sample image", files)
        chosen_file = os.path.join(SAMPLE_DIR, sample_name)
    else:
        st.sidebar.warning("No images found in sample_images/")

run_button = st.sidebar.button("▶ Run prediction")

# ---------------------------------------------------------------------
# 7. MAIN HEADER + HERO CARD
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">FHD-HYBRIDNET ALZHEIMER MRI</div>
        <p class="hero-subtitle">
            Three CNN backbones (<strong>DenseNet121</strong>, <strong>MobileNetV1</strong>, 
            <strong>ResNet50V2</strong>) combined with a fuzzy-logic ensemble using 
            <strong>Fuzzy Hellinger Distance</strong> for robust 4-class Alzheimer stage classification.
        </p>
        <div style="margin-top:0.6rem;">
            <span class="chip">🧠 <span>CNN ensemble</span></span>
            <span class="chip">📊 <span>Uncertainty-aware fusion</span></span>
            <span class="chip">🔥 <span>Grad-CAM interpretability</span></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br/>", unsafe_allow_html=True)

col_info, col_out = st.columns([1, 1])

with col_info:
    st.markdown(
        """
        <div class="info-card">
            <p style="margin:0 0 0.6rem 0; font-weight:650; color:#e5e7eb;">
                🔍 Workflow
            </p>
            <p style="margin:0 0 0.9rem 0; font-size:0.88rem; color:#d1d5db;">
                1) Choose a backbone or <strong>FHD-HybridNet</strong>,  
                2) Pick an MRI slice,  
                3) Hit <strong>Run prediction</strong> to view Grad-CAM and probabilities.
            </p>
            <p style="margin:0 0 0.45rem 0; font-weight:650; color:#e5e7eb;">
                Selected options
            </p>
        """,
        unsafe_allow_html=True,
    )
    st.write(f"**Model:** {model_name}")
    st.write(f"**Source:** {source}")
    if isinstance(chosen_file, str):
        st.write(f"**Image path:** `{chosen_file}`")
    elif chosen_file is None:
        st.write("_No image selected yet._")
    else:
        st.write(f"**Uploaded file:** `{chosen_file.name}`")
    st.markdown("</div>", unsafe_allow_html=True)

# If button not pressed, stop
if not run_button:
    st.stop()

if chosen_file is None:
    st.error("Please upload or select an image first.")
    st.stop()
# ===== WHITE THEME CSS =====
st.markdown("""
<style>

    /* GLOBAL BACKGROUND */
    .stApp {
        background: #fafafa !important;
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }

    /* CONTAINER WIDTH */
    .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }

    /* GENERAL TITLES */
    h1, h2, h3, h4 {
        color: #1e293b !important;
        font-weight: 700;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e293b !important;
    }

    section[data-testid="stSidebar"] label {
        font-weight: 600;
        color: #334155 !important;
    }

    /* CARDS */
    .hero-card {
        background: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }

    .info-card {
        background: #ffffff;
        border-radius: 1rem;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }

    .prediction-card {
        background: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem 1.7rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    }

    /* CHIPS */
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.78rem;
        margin-right: 6px;
    }

    /* BUTTONS */
    .stButton>button {
        background: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.2rem !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(37,99,235,0.35);
    }
    .stButton>button:hover {
        background: #1d4ed8 !important;
    }

    /* DOWNLOAD BUTTON */
    .stDownloadButton>button {
        background: #10b981 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.2rem !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stDownloadButton>button:hover {
        background: #059669 !important;
    }

    /* PLOT MARGIN */
    .stPyplot {
        margin-top: 0.5rem !important;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 8. LOAD IMAGE & RUN MODEL (UNCHANGED)
# ---------------------------------------------------------------------
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

with st.spinner("Running prediction…"):
    if model_name == "FHD-HybridNet":
        probs, pred_idx, chosen_key, grad_model = run_fhd_ensemble(batch)
        cam_title = "FHD-HybridNet"
    else:
        model = load_single_model(model_name)
        preds = model.predict(batch, verbose=0)[0]
        probs = preds
        pred_idx = int(np.argmax(probs))
        grad_model = model
        cam_title = model_name

pred_class = CLASS_NAMES[pred_idx]

with st.spinner("Computing Grad-CAM…"):
    heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
    overlay = overlay_heatmap_on_image(heatmap, orig_img, alpha=0.45)

# ---------------------------------------------------------------------
# 9. PREDICTION CARD WITH FIGURE
# ---------------------------------------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Prediction Output")

st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(orig_img)
axes[0].set_title(f"Original Image\nClass Name: {pred_class}")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title(f"Grad-CAM\n{cam_title}")
axes[1].axis("off")

axes[2].barh(CLASS_NAMES, probs, color="#38bdf8")
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Probability")
axes[2].set_title("Class Probabilities", fontsize=10, pad=6)
for i, cls in enumerate(CLASS_NAMES):
    axes[2].text(probs[i] + 0.01, i, f"{probs[i]:.3f}", va="center", color="#e5e7eb")
axes[2].invert_yaxis()
axes[2].tick_params(colors="#e5e7eb")
for spine in axes[2].spines.values():
    spine.set_color("#94a3b8")

plt.tight_layout()
st.pyplot(fig)

st.markdown(
    f"""
    <p class="final-pred">
        Final Prediction : <em>{pred_class}</em>
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 10. SAVE IMAGE BUTTON (SAME LOGIC, NEW STYLE)
# ---------------------------------------------------------------------
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
buf.seek(0)

st.download_button(
    "💾 Save result image",
    data=buf,
    file_name=f"result_{pred_class}.png",
    mime="image/png"
)

st.markdown("</div>", unsafe_allow_html=True)
