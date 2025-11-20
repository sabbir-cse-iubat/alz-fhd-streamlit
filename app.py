import os
import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 0. BASIC SETUP
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI Demo",
    layout="wide"
)

# Folder where models will be cached after first download
MODEL_DIR = "alz_fhd_models"
os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_DIR = "sample_images"
os.makedirs(SAMPLE_DIR, exist_ok=True)  # safe if already exists

# Image size used in your training
IMG_SIZE = (224, 224)

# Class names (must match training order)
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# ---------------------------------------------------------------------
# 1. GOOGLE DRIVE MODEL LINKS (YOUR LINKS)
# ---------------------------------------------------------------------
# You gave:
# densenet : https://drive.google.com/file/d/1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO/view?usp=drive_link
# mobilenet: https://drive.google.com/file/d/1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0/view?usp=drive_link
# resnet   : https://drive.google.com/file/d/1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW/view?usp=drive_link

DENSENET_ID  = "1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO"
MOBILENET_ID = "1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0"
RESNET_ID    = "1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW"


def gdrive_url(file_id: str) -> str:
    """Convert a Drive file id to a direct download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ---------------------------------------------------------------------
# 2. MODEL LOADING (CACHED)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    """
    Download (first time) + load the requested model.
    model_name ∈ {"DenseNet121", "MobileNetV1", "ResNet50V2"}
    """
    if model_name == "DenseNet121":
        file_name = "densenet_alz_fhd.keras"
        file_id = DENSENET_ID
    elif model_name == "MobileNetV1":
        file_name = "mobilenet_alz_fhd.keras"
        file_id = MOBILENET_ID
    elif model_name == "ResNet50V2":
        file_name = "resnet_alz_fhd.keras"
        file_id = RESNET_ID
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    local_path = os.path.join(MODEL_DIR, file_name)

    if not os.path.exists(local_path):
        url = gdrive_url(file_id)
        st.info(f"📥 Downloading {model_name} weights… (first time only)")
        local_path = tf.keras.utils.get_file(
            fname=file_name,
            origin=url,
            cache_dir=".",
            cache_subdir=MODEL_DIR
        )

    model = tf.keras.models.load_model(local_path, compile=False)
    return model


@st.cache_resource(show_spinner=False)
def load_all_base_models():
    """Load all three backbones once for FHD-HybridNet."""
    dn = load_single_model("DenseNet121")
    mb = load_single_model("MobileNetV1")
    rn = load_single_model("ResNet50V2")
    return {"DenseNet": dn, "MobileNet": mb, "ResNet": rn}


# ---------------------------------------------------------------------
# 3. IMAGE HELPERS
# ---------------------------------------------------------------------
def load_image_from_file(file, img_size=IMG_SIZE):
    """Load PIL image from uploaded file or path and resize."""
    if isinstance(file, str):
        img = Image.open(file).convert("RGB")
    else:
        img = Image.open(file).convert("RGB")
    img_resized = img.resize(img_size)
    arr = np.asarray(img_resized).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return img, batch


# ---------------------------------------------------------------------
# 4. GRAD-CAM HELPERS
# ---------------------------------------------------------------------
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D


def get_last_conv_layer_name(model):
    """Try to find a suitable last convolution layer for Grad-CAM."""
    conv_types = (Conv2D, DepthwiseConv2D, SeparableConv2D)

    # First: look for explicit conv-like layers
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
        # Unwrap nested models / Sequential inside Functional
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, conv_types):
                    return sub.name

    # Fallback: any 4D output layer
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
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # (H, W, C)

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
# 5. FHD ENSEMBLE HELPERS
# ---------------------------------------------------------------------
def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd_single(models_dict, img_batch):
    """
    Run all three backbones, compute FHD between their probability vectors,
    and choose the backbone with minimal average distance (max consensus).
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
    return chosen_probs, pred_idx, chosen_key


def run_fhd_ensemble(img_batch):
    """Wrapper for FHD-HybridNet."""
    models_dict = load_all_base_models()
    probs, pred_idx, chosen_key = ensemble_predict_fhd_single(models_dict, img_batch)
    grad_model = models_dict[chosen_key]
    return probs, pred_idx, chosen_key, grad_model


# ---------------------------------------------------------------------
# 6. SIDEBAR CONTROLS
# ---------------------------------------------------------------------
st.sidebar.title("Controls")

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
    # create dummy list if folder empty to avoid error
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

# Placeholder for save button later
save_placeholder = st.sidebar.empty()

# ---------------------------------------------------------------------
# 7. MAIN AREA HEADER
# ---------------------------------------------------------------------
st.title("FHD-HybridNet Alzheimer MRI Classification Demo")
st.markdown(
    """
This app uses three CNN backbones (**DenseNet121, MobileNetV1, ResNet50V2**)  
and a fuzzy-logic-based ensemble (**Fuzzy Hellinger Distance**)  
to classify Alzheimer MRI scans into **four classes**.
"""
)

# ---------------------------------------------------------------------
# 8. MAIN LOGIC
# ---------------------------------------------------------------------
col_info, col_out = st.columns([1, 2])

with col_info:
    st.subheader("Selected options")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Source:** {source}")
    if isinstance(chosen_file, str):
        st.write(f"**Image path:** `{chosen_file}`")
    elif chosen_file is None:
        st.write("_No image selected yet._")
    else:
        st.write(f"**Uploaded file:** `{chosen_file.name}`")

if not run_button:
    with col_out:
        st.info("👈 Select a model & image, then click **Run prediction**.")
    st.stop()

if chosen_file is None:
    st.error("Please upload or select an image first.")
    st.stop()

# Load image & batch
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

# Run model
with st.spinner("Running prediction…"):
    if model_name == "FHD-HybridNet":
        probs, pred_idx, chosen_key, grad_model = run_fhd_ensemble(batch)
        cam_title = f"FHD-HybridNet ({chosen_key})"
    else:
        model = load_single_model(model_name)
        preds = model.predict(batch, verbose=0)[0]
        probs = preds
        pred_idx = int(np.argmax(probs))
        grad_model = model
        cam_title = model_name

pred_class = CLASS_NAMES[pred_idx]

# Grad-CAM
with st.spinner("Computing Grad-CAM…"):
    heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
    overlay = overlay_heatmap_on_image(heatmap, orig_img, alpha=0.45)

# ---------------------------------------------------------------------
# 9. VISUAL OUTPUT + SAVE BUTTON
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original
axes[0].imshow(orig_img)
axes[0].set_title(f"Original\nPredicted: {pred_class}")
axes[0].axis("off")

# Grad-CAM
axes[1].imshow(overlay)
axes[1].set_title(f"Grad-CAM\n{cam_title}")
axes[1].axis("off")

# Probabilities
axes[2].barh(CLASS_NAMES, probs)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Probability")
axes[2].set_title("Class probabilities")
for i, cls in enumerate(CLASS_NAMES):
    axes[2].text(
        probs[i] + 0.01,
        i,
        f"{probs[i]:.3f}",
        va="center"
    )

plt.tight_layout()

with col_out:
    st.pyplot(fig)

    st.markdown(f"### Predicted class: **{pred_class}**")
    st.markdown("#### Probabilities")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"- **{cls:16s}**: {probs[i]:.4f}")

# Save result image button
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
buf.seek(0)

save_placeholder.download_button(
    "💾 Save result image",
    data=buf,
    file_name=f"result_{pred_class}.png",
    mime="image/png"
)
