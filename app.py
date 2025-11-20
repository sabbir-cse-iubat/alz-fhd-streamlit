import os
import io
import warnings
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# --------------------------------------------------------------------
# BASIC CONFIG
# --------------------------------------------------------------------
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
IMG_SIZE = (224, 224)

MODEL_CACHE_DIR = "alz_fhd_models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

if "last_result_png" not in st.session_state:
    st.session_state["last_result_png"] = None

# --------------------------------------------------------------------
# GOOGLE DRIVE MODEL IDs (YOUR TRAINED MODELS)
# --------------------------------------------------------------------
# densenet : https://drive.google.com/file/d/1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO/view?usp=drive_link
# mobilenet: https://drive.google.com/file/d/1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0/view?usp=drive_link
# resnet   : https://drive.google.com/file/d/1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW/view?usp=drive_link

DENSENET_ID  = "1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO"
MOBILENET_ID = "1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0"
RESNET_ID    = "1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW"


def gdrive_direct_url(file_id: str) -> str:
    """Direct download URL usable by gdown."""
    return f"https://drive.google.com/uc?id={file_id}"

# --------------------------------------------------------------------
# GDOWN DOWNLOADER — ONLY THIS, NO tf.keras.utils.get_file
# --------------------------------------------------------------------
def _download_model_if_needed(file_id: str, filename: str) -> str:
    """
    Use gdown to download from Google Drive into MODEL_CACHE_DIR.
    Returns the local file path. Only downloads once.
    """
    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    # Already downloaded
    if os.path.exists(local_path):
        return local_path

    url = gdrive_direct_url(file_id)
    st.sidebar.info(f"📥 Downloading model: {filename} (first time only)…")

    try:
        gdown.download(url, local_path, quiet=False)
    except Exception as e:
        st.sidebar.error(f"❌ Failed to download {filename} from Google Drive.\n\n{e}")
        raise

    if not os.path.exists(local_path):
        # Download failed or file not created (e.g. wrong sharing / non-public file)
        raise FileNotFoundError(
            f"Download seems to have failed; file not found at {local_path}"
        )

    return local_path

# --------------------------------------------------------------------
# SINGLE MODEL LOADER (Keras)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    """
    Download (if needed) + load the requested model.
    model_name ∈ {"DenseNet121", "MobileNetV1", "ResNet50V2"}
    """
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
            "❌ Failed to load model from file.\n\n"
            f"Path: `{local_path}`\n"
            "Common causes:\n"
            "• File is not actually a `.keras` model (e.g. zipped or HTML)\n"
            "• Upload/training saved in a different format\n\n"
            f"Raw error:\n{e}"
        )
        raise

    return model
# --------------------------------------------------------------------
# ALL BASE MODELS LOADER (for FHD-HybridNet)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_all_base_models():
    """
    Load all three backbones once and reuse them.
    Keys are the exact backbone names used in ensemble_predict_fhd_single.
    """
    dn = load_single_model("DenseNet121")
    mb = load_single_model("MobileNetV1")
    rn = load_single_model("ResNet50V2")

    return {
        "DenseNet121": dn,
        "MobileNetV1": mb,
        "ResNet50V2": rn,
    }

# --------------------------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------------------------
def load_image(path: str):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return img, arr

# --------------------------------------------------------------------
# GRAD-CAM UTILITIES
# --------------------------------------------------------------------
def get_last_conv_layer_name(model):
    conv_types = (Conv2D, DepthwiseConv2D, SeparableConv2D)

    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No suitable conv layer found in model.")


def make_gradcam_heatmap(model, img_array, class_index):
    last_conv_name = get_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        conv_outputs = conv_outputs[0]          # (H, W, C)
        preds = preds[0]                        # (num_classes,)
        class_channel = preds[class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (C,)

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, pil_img, alpha=0.45):
    w, h = pil_img.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLOR_BGR2RGB)
    base = np.asarray(pil_img).astype("float32") / 255.0

    overlay = alpha * (heatmap_color.astype("float32") / 255.0) + (1 - alpha) * base
    return np.clip(overlay, 0.0, 1.0)

# --------------------------------------------------------------------
# FHD ENSEMBLE
# --------------------------------------------------------------------
def ensemble_predict_fhd_single(img_batch):
    models_dict = load_all_base_models()
    keys = ["DenseNet121", "MobileNetV1", "ResNet50V2"]

    preds_list = []
    for k in keys:
        preds_list.append(models_dict[k].predict(img_batch, verbose=0)[0])

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
    return pred_idx, chosen_probs, chosen_key


# --------------------------------------------------------------------
# SIDEBAR UI (AS BEFORE)
# --------------------------------------------------------------------
st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox(
    "Select model",
    ["FHD-HybridNet", "DenseNet121", "MobileNetV1", "ResNet50V2"],
)

source_choice = st.sidebar.radio(
    "Choose image source",
    ["Upload MRI", "Sample gallery"],
)

if source_choice == "Sample gallery":
    if not os.path.isdir("sample_images"):
        st.sidebar.error("sample_images/ folder not found in repo.")
        sample_files = []
        selected_image_name = None
        image_path = None
    else:
        sample_files = sorted(os.listdir("sample_images"))
        sample_files = [f for f in sample_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        selected_image_name = st.sidebar.selectbox("Pick a sample image", sample_files)
        image_path = (
            os.path.join("sample_images", selected_image_name)
            if selected_image_name
            else None
        )
else:
    uploaded = st.sidebar.file_uploader("Upload MRI", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        temp_path = os.path.join("/tmp", uploaded.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        image_path = temp_path
        selected_image_name = uploaded.name
    else:
        image_path = None
        selected_image_name = None

btn_predict = st.sidebar.button("Predict", use_container_width=True)
btn_save    = st.sidebar.button("Save result", use_container_width=True)

# --------------------------------------------------------------------
# MAIN PAGE TEXT
# --------------------------------------------------------------------
st.title("FHD-HybridNet Alzheimer MRI Classifier")

st.write(
    "This app uses three CNN backbones (**DenseNet121**, **MobileNetV1**, "
    "**ResNet50V2**) and a fuzzy-logic-based ensemble "
    "(**Fuzzy Hellinger Distance**) to classify Alzheimer MRI scans "
    "into four classes."
)

# The info box placed exactly here
st.info("Select a model & image, then click **Predict**.")

# Selected-options block
st.subheader("Selected options")
st.write(f"**Model:** {model_choice}")
st.write(f"**Source:** {source_choice}")
st.write(f"**Image path:** {image_path if image_path else 'None'}")

st.markdown("---")

# Placeholder for download button / messages
download_placeholder = st.empty()

# --------------------------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------------------------
if btn_predict:
    if not image_path or not os.path.exists(image_path):
        st.error("No valid image selected.")
    else:
        with st.spinner("Loading image..."):
            pil_img, img_batch = load_image(image_path)

        with st.spinner("Running prediction..."):
            if model_choice == "FHD-HybridNet":
                probs, pred_idx, chosen_key, grad_model = ensemble_predict_fhd_single(
                    img_batch
                )
                cam_title = f"FHD-HybridNet ({chosen_key})"
            else:
                grad_model = load_single_model(model_choice)
                preds = grad_model.predict(img_batch, verbose=0)[0]
                probs = preds
                pred_idx = int(np.argmax(probs))
                cam_title = model_choice

        pred_class_name = CLASS_NAMES[pred_idx]

        with st.spinner("Computing Grad-CAM..."):
            heatmap = make_gradcam_heatmap(grad_model, img_batch, class_index=pred_idx)
            overlay = overlay_heatmap_on_image(heatmap, pil_img, alpha=0.45)

        # ----- Plot nicely (similar to your research UI) -----
        fig = plt.figure(figsize=(15, 6))

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(pil_img)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(overlay)
        ax2.set_title(f"Grad-CAM\n{cam_title}")
        ax2.axis("off")

        ax3 = plt.subplot(1, 3, 3)
        bars = ax3.barh(CLASS_NAMES, probs)
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability")
        ax3.set_title("Class probabilities")
        for i, bar in enumerate(bars):
            ax3.text(
                probs[i] + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{probs[i]:.3f}",
                va="center",
            )

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader(f"Predicted Class: **{pred_class_name}**")

        # Save figure to session as PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.session_state["last_result_png"] = buf.read()

        download_placeholder.empty()
        download_placeholder.success(
            "Result ready. Use **Save result** from the sidebar to download."
        )

# --------------------------------------------------------------------
# SAVE RESULT HANDLER
# --------------------------------------------------------------------
if btn_save:
    img_bytes = st.session_state.get("last_result_png", None)
    if img_bytes is None:
        download_placeholder.error("No result to save. Run **Predict** first.")
    else:
        download_placeholder.download_button(
            label="⬇️ Download last prediction as PNG",
            data=img_bytes,
            file_name="fhd_hybridnet_result.png",
            mime="image/png",
        )
