import os
import io
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI Classifier",
    layout="wide"
)

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

IMG_SIZE = (224, 224)
MODEL_DIR = "alz_fhd_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive links (your links)
DENSENET_URL = (
    "https://drive.google.com/uc?export=download&id=1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO"
)
MOBILENET_URL = (
    "https://drive.google.com/uc?export=download&id=1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0"
)
RESNET_URL = (
    "https://drive.google.com/uc?export=download&id=1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW"
)


def get_url_and_fname(backbone: str):
    """Return (url, local_filename) for each backbone."""
    if backbone == "DenseNet121":
        return DENSENET_URL, os.path.join(MODEL_DIR, "densenet_alz_fhd.keras")
    if backbone == "MobileNetV1":
        return MOBILENET_URL, os.path.join(MODEL_DIR, "mobilenet_alz_fhd.keras")
    if backbone == "ResNet50V2":
        return RESNET_URL, os.path.join(MODEL_DIR, "resnet_alz_fhd.keras")
    raise ValueError(f"Unknown backbone: {backbone}")


# ---------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_single_model(backbone_name: str):
    """
    Download (if needed) and load a single keras model for the given backbone.
    """
    url, local_path = get_url_and_fname(backbone_name)

    # Download only if file not present
    if not os.path.exists(local_path):
        with st.spinner(f"Downloading {backbone_name} weights… (first time only)"):
            tf.keras.utils.get_file(
                fname=os.path.basename(local_path),
                origin=url,
                cache_dir=".",
                cache_subdir=MODEL_DIR,
            )
    model = tf.keras.models.load_model(local_path, compile=False)
    return model


@st.cache_resource(show_spinner=False)
def load_all_base_models():
    """
    Load all three base models once and reuse (for FHD-HybridNet).
    """
    dn = load_single_model("DenseNet121")
    mb = load_single_model("MobileNetV1")
    rn = load_single_model("ResNet50V2")
    return {"DenseNet121": dn, "MobileNetV1": mb, "ResNet50V2": rn}


# ---------------------------------------------------------
# Image utilities
# ---------------------------------------------------------


def load_image_to_array(file_obj_or_path, target_size=IMG_SIZE):
    """
    Load image from either a file-like or a filesystem path,
    return (PIL_image, np_array_batch).
    """
    if hasattr(file_obj_or_path, "read"):
        img = Image.open(file_obj_or_path).convert("RGB")
    else:
        img = Image.open(str(file_obj_or_path)).convert("RGB")

    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return img, arr


# ---------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------


from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D


def get_last_conv_layer_name(model):
    """
    Try to find the last 4D conv-like layer.
    """
    conv_types = (Conv2D, DepthwiseConv2D, SeparableConv2D)

    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
        # Nested models / Sequential inside Functional
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, conv_types):
                    return sub.name

    # fallback: check by output_shape
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

        # Some models return lists of tensors
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if class_index is None:
            class_index = tf.argmax(preds[0])

        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap, pil_image, alpha=0.45):
    img = pil_image
    w, h = img.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base = np.asarray(img).astype("float32") / 255.0
    overlay = alpha * (heatmap_color.astype("float32") / 255.0) + (1 - alpha) * base
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


# ---------------------------------------------------------
# Fuzzy Hellinger Distance ensemble
# ---------------------------------------------------------


def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd_single(models_dict, img_batch):
    """
    Run all three base models, compute FHD among their probability vectors,
    and choose the model whose predictions are most in consensus.
    Returns:
      probs      : np.array (num_classes,)
      pred_idx   : int
      chosen_key : str (one of the backbone names)
    """
    keys = ["DenseNet121", "MobileNetV1", "ResNet50V2"]
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


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

# ---- Header text ----
st.title("FHD-HybridNet Alzheimer MRI Classifier")

st.write(
    "This app uses three CNN backbones (**DenseNet121**, **MobileNetV1**, "
    "**ResNet50V2**) and a fuzzy-logic-based ensemble (**Fuzzy Hellinger "
    "Distance**) to classify Alzheimer MRI scans into four classes."
)

# Instruction box (your requested line)
st.info("Select a model & image, then click **Run prediction**.")

st.markdown("---")

# ---- Sidebar controls ----
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select model",
    ["FHD-HybridNet", "DenseNet121", "MobileNetV1", "ResNet50V2"],
    index=0,
)

image_mode = st.sidebar.radio(
    "Choose image source",
    ["Upload MRI", "Sample gallery"],
    index=1,
)

uploaded_file = None
sample_file_name = None
sample_path_for_display = None

SAMPLE_DIR = "sample_images"

if image_mode == "Upload MRI":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an MRI image",
        type=["png", "jpg", "jpeg"],
    )
else:
    # Sample gallery from repo folder
    sample_files = []
    if os.path.isdir(SAMPLE_DIR):
        for f in sorted(os.listdir(SAMPLE_DIR)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                sample_files.append(f)

    if not sample_files:
        st.sidebar.warning(
            "No sample images found in `sample_images/` folder in the repo."
        )
    else:
        sample_file_name = st.sidebar.selectbox(
            "Pick a sample image", sample_files, index=0
        )
        sample_path_for_display = f"{SAMPLE_DIR}/{sample_file_name}"

# Run button on main page (not sidebar, as you requested a clear action)
run_prediction = st.button("Run prediction", type="primary")

# Placeholder for final figure for download
combined_fig = None

# ---------------------------------------------------------
# Selected options box (just after description and before results)
# ---------------------------------------------------------

def get_display_image_path():
    if image_mode == "Upload MRI":
        if uploaded_file is not None:
            return uploaded_file.name
        return "(no file uploaded)"
    else:
        if sample_path_for_display is not None:
            return sample_path_for_display
        return "(no sample selected)"


st.subheader("Selected options")
st.write(f"**Model:** {model_name}")
st.write(f"**Source:** {image_mode}")
st.write(f"**Image path:** {get_display_image_path()}")

st.markdown("---")

# ---------------------------------------------------------
# Core prediction logic (triggered by button)
# ---------------------------------------------------------

if run_prediction:
    if image_mode == "Upload MRI" and uploaded_file is None:
        st.error("Please upload an MRI image first.")
    elif image_mode == "Sample gallery" and sample_file_name is None:
        st.error("No sample image available / selected.")
    else:
        # 1. Load image
        if image_mode == "Upload MRI":
            img_pil, batch = load_image_to_array(uploaded_file, IMG_SIZE)
        else:
            img_path = os.path.join(SAMPLE_DIR, sample_file_name)
            img_pil, batch = load_image_to_array(img_path, IMG_SIZE)

        # 2. Run model
        with st.spinner("Running prediction…"):
            if model_name == "FHD-HybridNet":
                probs, pred_idx, chosen_key, grad_model = run_fhd_ensemble(batch)
                cam_title = f"FHD-HybridNet ({chosen_key})"
            else:
                model = load_single_model(model_name)
                preds = model.predict(batch, verbose=0)[0]
                probs = preds
                pred_idx = int(np.argmax(preds))
                grad_model = model
                cam_title = model_name

        pred_class = CLASS_NAMES[pred_idx]

        # 3. Grad-CAM
        with st.spinner("Computing Grad-CAM…"):
            heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
            overlay = overlay_heatmap_on_image(heatmap, img_pil, alpha=0.45)

        # 4. Plot results
        fig = plt.figure(figsize=(14, 5))

        # Original
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(img_pil)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Grad-CAM
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(overlay)
        ax2.set_title(f"Grad-CAM\n{cam_title}")
        ax2.axis("off")

        # Probabilities
        ax3 = plt.subplot(1, 3, 3)
        y_pos = np.arange(len(CLASS_NAMES))
        bars = ax3.barh(y_pos, probs, align="center")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(CLASS_NAMES)
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
        st.markdown(f"### Predicted Class: **{pred_class}**")

        # Save figure to buffer for download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        combined_fig = buf

        # Download button
        st.download_button(
            label="Download result image",
            data=combined_fig,
            file_name=f"alz_fhd_prediction_{pred_class}.png",
            mime="image/png",
        )
