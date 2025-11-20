import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import streamlit as st

# -------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

SAMPLE_DIR = "sample_images"

# === TODO: এখানকার তিনটা URL তোমার নিজের model (.keras) ফাইলের direct link দিও ===
DENSENET_URL = "https://drive.google.com/file/d/1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO/view?usp=drive_link"
MOBILENET_URL = "https://drive.google.com/file/d/1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0/view?usp=drive_link"
RESNET_URL = "https://drive.google.com/file/d/1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW/view?usp=drive_link"


# -------------------------------------------------------------------
# Utility: download + cache keras models
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_single_model(backbone_name: str):
    """
    Download (if needed) and load a single keras model.
    backbone_name in {"DenseNet121", "MobileNetV1", "ResNet50V2"}.
    """
    if backbone_name == "DenseNet121":
        url = DENSENET_URL
        fname = "densenet_alz_fhd.keras"
    elif backbone_name == "MobileNetV1":
        url = MOBILENET_URL
        fname = "mobilenet_alz_fhd.keras"
    elif backbone_name == "ResNet50V2":
        url = RESNET_URL
        fname = "resnet_alz_fhd.keras"
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Download file into a stable local cache (inside Streamlit environment)
    local_path = tf.keras.utils.get_file(
        fname=fname,
        origin=url,
        cache_subdir="alz_fhd_models",
        cache_dir=".",
    )

    model = tf.keras.models.load_model(local_path, compile=False)
    return model


@st.cache_resource(show_spinner=False)
def load_all_base_models():
    """
    Load all three base models once and reuse (for FHD-HybridNet).
    Returns a dict: {"DenseNet": model, "MobileNet": model, "ResNet": model}
    """
    dn = load_single_model("DenseNet121")
    mb = load_single_model("MobileNetV1")
    rn = load_single_model("ResNet50V2")
    return {
        "DenseNet": dn,
        "MobileNet": mb,
        "ResNet": rn,
    }


# -------------------------------------------------------------------
# Image preprocessing
# -------------------------------------------------------------------
def preprocess_image_array(img_np: np.ndarray) -> np.ndarray:
    """
    img_np: H x W x 3, uint8
    return: 1 x H x W x 3 float32 in [0,1]
    """
    img_resized = cv2.resize(img_np, IMG_SIZE)
    img_resized = img_resized.astype("float32") / 255.0
    batch = np.expand_dims(img_resized, axis=0)
    return batch


# -------------------------------------------------------------------
# Grad-CAM helpers (Keras 3 / TF 2.20 safe)
# -------------------------------------------------------------------
def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Return name of the last layer whose output is 4D (batch, H, W, C).
    """
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            continue

        if isinstance(out_shape, (tuple, list)) and len(out_shape) == 4:
            return layer.name

    raise ValueError("No suitable conv layer found in model.")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int | None = None,
) -> np.ndarray:
    """
    img_array: shape (1, H, W, 3)
    """
    last_conv_name = get_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam_on_image(
    heatmap: np.ndarray,
    pil_image: Image.Image,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Returns RGB float32 image in [0,1] with Grad-CAM overlay.
    """
    img = np.array(pil_image).astype("float32") / 255.0
    h, w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = heatmap_color.astype("float32") / 255.0

    overlay = alpha * heatmap_color + (1.0 - alpha) * img
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


# -------------------------------------------------------------------
# Fuzzy Hellinger Distance ensemble (FHD-HybridNet)
# -------------------------------------------------------------------
def fuzzy_hellinger_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd_single(models_dict: dict, img_batch: np.ndarray):
    """
    models_dict: {"DenseNet": model, "MobileNet": model, "ResNet": model}
    img_batch: shape (1, H, W, 3)

    returns:
        pred_idx: int
        chosen_probs: np.ndarray, shape (NUM_CLASSES,)
        chosen_key: "DenseNet" / "MobileNet" / "ResNet"
    """
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
    return pred_idx, chosen_probs, chosen_key


# Wrapper used by main app for ensemble
def run_fhd_ensemble(img_batch: np.ndarray):
    """
    Returns:
        probs: np.ndarray (NUM_CLASSES,)
        pred_idx: int
        chosen_key: str (DenseNet/MobileNet/ResNet)
        grad_model: keras.Model (the selected backbone)
    """
    models_dict = load_all_base_models()
    pred_idx, probs, chosen_key = ensemble_predict_fhd_single(models_dict, img_batch)
    grad_model = models_dict[chosen_key]
    return probs, pred_idx, chosen_key, grad_model


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI Classifier",
    layout="wide",
)

st.title("🧠 FHD-HybridNet Alzheimer MRI Classifier")

st.write(
    "This app uses three CNN backbones (DenseNet121, MobileNetV1, ResNet50V2) "
    "and a fuzzy-logic-based ensemble (**Fuzzy Hellinger Distance**) to classify "
    "Alzheimer MRI scans into four classes."
)

st.markdown("---")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select model",
    ["DenseNet121", "MobileNetV1", "ResNet50V2", "FHD-HybridNet"],
    index=3,
)

source = st.sidebar.radio(
    "Choose image source",
    ["Upload MRI", "Sample gallery"],
    index=1,
)

selected_file = None
uploaded_file = None

if source == "Upload MRI":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an MRI image", type=["png", "jpg", "jpeg"]
    )
else:
    if not os.path.isdir(SAMPLE_DIR):
        st.sidebar.warning(
            f"No '{SAMPLE_DIR}' folder found. "
            "Create it in the repo and add some .png/.jpg images."
        )
        sample_files = []
    else:
        sample_files = sorted(
            f
            for f in os.listdir(SAMPLE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    if sample_files:
        selected_file = st.sidebar.selectbox("Pick a sample image", sample_files)
    else:
        selected_file = None
        st.sidebar.info("No sample images found in the folder.")

run_button = st.sidebar.button("🔍 Predict")


# ---------------- Main prediction logic ----------------
if run_button:
    # 1. Resolve image
    if source == "Upload MRI":
        if uploaded_file is None:
            st.error("Please upload an MRI image first.")
            st.stop()
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        img_batch = preprocess_image_array(img_np)
        img_path_str = uploaded_file.name
    else:
        if not selected_file:
            st.error("No sample image selected.")
            st.stop()
        img_path = os.path.join(SAMPLE_DIR, selected_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_batch = preprocess_image_array(img_np)
        img_path_str = img_path

    # 2. Run model
    with st.spinner("Running prediction..."):
        if model_name == "FHD-HybridNet":
            probs, pred_idx, chosen_key, grad_model = run_fhd_ensemble(img_batch)
            title_cam = f"FHD-HybridNet ({chosen_key})"
        else:
            model = load_single_model(model_name)
            preds = model.predict(img_batch, verbose=0)[0]
            probs = preds
            pred_idx = int(np.argmax(preds))
            grad_model = model
            title_cam = model_name

        pred_class = CLASS_NAMES[pred_idx]

        # Grad-CAM
        heatmap = make_gradcam_heatmap(grad_model, img_batch, class_index=pred_idx)
        overlay = overlay_gradcam_on_image(heatmap, img)

    # 3. Draw figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(img)
    axes[0].set_title(f"Original\n({os.path.basename(img_path_str)})")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM\n{title_cam}")
    axes[1].axis("off")

    # Probabilities bar chart
    y_pos = np.arange(len(CLASS_NAMES))
    axes[2].barh(y_pos, probs, color="#1f77b4")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(CLASS_NAMES)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Probability")
    axes[2].set_title("Class probabilities")

    for i, p in enumerate(probs):
        axes[2].text(
            p + 0.01,
            i,
            f"{p:.3f}",
            va="center",
        )

    plt.tight_layout()

    # Show in app
    st.pyplot(fig)

    st.markdown(f"### ✅ Predicted class: `{pred_class}`")

    # 4. Download result as single image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)

    st.download_button(
        label="💾 Save result as image",
        data=buf,
        file_name="fhd_prediction_result.png",
        mime="image/png",
    )

else:
    st.info("Select a model, choose an image, then click **🔍 Predict** on the left.")
