import os
import io
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import streamlit as st
import gdown

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# ---------------------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI Classifier",
    layout="wide",
)

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
IMG_SIZE = (224, 224)

MODEL_CACHE_DIR = "alz_fhd_models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. GOOGLE DRIVE MODEL LINKS (YOUR LINKS -> IDs)
# ---------------------------------------------------------------------
# densenet : https://drive.google.com/file/d/1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO/view?usp=drive_link
# mobilenet: https://drive.google.com/file/d/1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0/view?usp=drive_link
# resnet   : https://drive.google.com/file/d/1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW/view?usp=drive_link

DENSENET_ID  = "1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO"
MOBILENET_ID = "1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0"
RESNET_ID    = "1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW"


def gdrive_direct_url(file_id: str) -> str:
    """Direct download URL for gdown."""
    return f"https://drive.google.com/uc?id={file_id}"


# ---------------------------------------------------------------------
# 2. MODEL LOADING VIA GDOWN (NO .keras/datasets PATH)
# ---------------------------------------------------------------------
def _download_model_if_needed(file_id: str, filename: str) -> str:
    """
    Use gdown to download from Google Drive into MODEL_CACHE_DIR.
    Returns the local file path.
    """
    local_path = os.path.join(MODEL_CACHE_DIR, filename)

    if not os.path.exists(local_path):
        url = gdrive_direct_url(file_id)
        st.info(f"📥 Downloading model: {filename} (first time only)")
        try:
            gdown.download(url, local_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download {filename} from Google Drive.\nError: {e}")
            raise

    if not os.path.exists(local_path):
        st.error(f"Model file {filename} was not created. Check Drive sharing / ID.")
        raise FileNotFoundError(local_path)

    return local_path


@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    """
    Download (first time) + load the requested model using gdown.
    Accepts either: "DenseNet121"/"DenseNet", "MobileNetV1"/"MobileNet",
    "ResNet50V2"/"ResNet".
    """
    if model_name in ("DenseNet121", "DenseNet"):
        file_id = DENSENET_ID
        fname = "densenet_alz_fhd.keras"
    elif model_name in ("MobileNetV1", "MobileNet"):
        file_id = MOBILENET_ID
        fname = "mobilenet_alz_fhd.keras"
    elif model_name in ("ResNet50V2", "ResNet"):
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
    """Load all three backbones once for FHD-HybridNet."""
    dn = load_single_model("DenseNet")
    mb = load_single_model("MobileNet")
    rn = load_single_model("ResNet")
    return {"DenseNet": dn, "MobileNet": mb, "ResNet": rn}


# ---------------------------------------------------------------------
# 3. IMAGE UTILS
# ---------------------------------------------------------------------
def load_image_to_array(img_path: str, target_size=IMG_SIZE):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return img, arr


# ---------------------------------------------------------------------
# 4. GRAD-CAM
# ---------------------------------------------------------------------
def get_last_conv_layer_name(model):
    """
    Find the last conv-like layer (Conv2D / DepthwiseConv2D / SeparableConv2D).
    Falls back to the last 4D-output layer if needed.
    """
    conv_types = (Conv2D, DepthwiseConv2D, SeparableConv2D)

    # Try explicit conv layers
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, conv_types):
                    return sub.name

    # Fallback: any 4D layer
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
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if class_index is None:
            class_index = tf.argmax(preds[0])

        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, pil_image, alpha=0.45):
    w, h = pil_image.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base = np.asarray(pil_image).astype("float32") / 255.0
    overlay = alpha * (heatmap_color.astype("float32") / 255.0) + (1 - alpha) * base
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


# ---------------------------------------------------------------------
# 5. FHD ENSEMBLE
# ---------------------------------------------------------------------
def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd_single(models_dict, img_batch):
    keys = ["DenseNet", "MobileNet", "ResNet"]
    preds_list = []

    for k in keys:
        preds_list.append(models_dict[k].predict(img_batch, verbose=0)[0])

    m_count = len(preds_list)
    avg_fhd = []

    for i in range(m_count):
        dists = []
        for j in range(m_count):
            if i == j:
                continue
            dists.append(fuzzy_hellinger_distance(preds_list[i], preds_list[j]))
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
# 6. UI LAYOUT
# ---------------------------------------------------------------------
st.title("FHD-HybridNet Alzheimer MRI Classifier")

st.write(
    "This app uses three CNN backbones (**DenseNet121**, **MobileNetV1**, "
    "**ResNet50V2**) and a fuzzy-logic-based ensemble "
    "(**Fuzzy Hellinger Distance**) to classify Alzheimer MRI scans into "
    "four classes: Mild, Moderate, Non, and Very Mild Dementia."
)

# Helper info box
st.info("Select a model & image, then click **Predict**.")

# Controls row (like your Colab UI)
col_model, col_path = st.columns([1, 3])

with col_model:
    selected_model = st.selectbox(
        "Model",
        ["FHD-HybridNet", "DenseNet", "MobileNet", "ResNet"],
        index=0,
    )

with col_path:
    default_path = "sample_images/MildDemented_11.jpg"
    image_path = st.text_input("Image", value=default_path)

col_btn_pred, col_btn_save = st.columns([1, 1])
predict_clicked = col_btn_pred.button("Predict")
save_placeholder = col_btn_save.empty()

# Selected options block
st.subheader("Selected options")
st.write(f"**Model:** {selected_model}")
st.write("**Source:** File path from repository")
st.write(f"**Image path:** {image_path}")

st.markdown("---")

combined_fig_buf = None

# ---------------------------------------------------------------------
# 7. PREDICTION + VISUALIZATION
# ---------------------------------------------------------------------
if predict_clicked:
    if not image_path or not os.path.exists(image_path):
        st.error(
            "Image path not found on server. "
            "Make sure the image exists in the repo (e.g. `sample_images/...`)."
        )
    else:
        # 1) Load image
        with st.spinner("Loading image..."):
            orig_img, batch = load_image_to_array(image_path, IMG_SIZE)

        # 2) Run model
        with st.spinner("Running prediction..."):
            if selected_model == "FHD-HybridNet":
                probs, pred_idx, chosen_key, grad_model = run_fhd_ensemble(batch)
                cam_title = f"FHD-HybridNet ({chosen_key})"
            else:
                model = load_single_model(selected_model)
                preds = model.predict(batch, verbose=0)[0]
                probs = preds
                pred_idx = int(np.argmax(preds))
                grad_model = model
                cam_title = selected_model

        pred_class_name = CLASS_NAMES[pred_idx]

        # 3) Grad-CAM
        with st.spinner("Computing Grad-CAM..."):
            heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
            overlay = overlay_heatmap_on_image(heatmap, orig_img, alpha=0.45)

        # 4) Plot like your research UI
        fig = plt.figure(figsize=(14, 6))

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(orig_img)
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(overlay)
        ax2.set_title(f"Predicted Image (Grad-CAM)\n{cam_title}")
        ax2.axis("off")

        ax3 = plt.subplot(1, 3, 3)
        bars = ax3.barh(CLASS_NAMES, probs)
        ax3.set_xlim(0, 1.0)
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

        st.markdown(f"### Predicted Class: **{pred_class_name}**")

        # 5) Prepare for download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        combined_fig_buf = buf

        save_placeholder.download_button(
            label="Save result",
            data=combined_fig_buf,
            file_name=f"fhd_prediction_{pred_class_name}.png",
            mime="image/png",
        )
