import os
import numpy as np
import tensorflow as tf
import cv2
import gdown
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# 1. Basic config
# ==========================

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
IMG_SIZE = (224, 224)

# ---- TODO: PUT YOUR GOOGLE DRIVE FILE IDS HERE ----
DENSENET_ID  = "1jPnrxTcTI8WW7d1Xwcudf40k4tf6SIXO"
MOBILENET_ID = "1eQLv6ruj64RAxiXQ9Vl-fUwkzXeYl8m0"
RESNET_ID    = "1tnyMBE16BEBSBBx5jKQdzWNTxr1huBcW"
# ---------------------------------------------------

FILE_IDS = {
    "DenseNet": DENSENET_ID,
    "MobileNet": MOBILENET_ID,
    "ResNet": RESNET_ID,
}

LOCAL_MODEL_PATHS = {
    "DenseNet": "models/densenet_alz_fhd.keras",
    "MobileNet": "models/mobilenet_alz_fhd.keras",
    "ResNet": "models/resnet_alz_fhd.keras",
}

os.makedirs("models", exist_ok=True)

# ==========================
# 2. Model download & loading
# ==========================

def download_model_if_needed(name: str) -> str:
    """
    If local model file doesn't exist, download it from Google Drive.
    Returns local path.
    """
    file_id = FILE_IDS[name]
    local_path = LOCAL_MODEL_PATHS[name]

    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write(f"📥 Downloading {name} weights… (first time only)")
        gdown.download(url, local_path, quiet=False)
    return local_path


@st.cache_resource
def load_tf_model(name: str):
    local_path = download_model_if_needed(name)
    model = tf.keras.models.load_model(local_path)
    return model

# ==========================
# 3. Image pre-processing
# ==========================

def preprocess_image(pil_img, img_size=IMG_SIZE):
    """
    Resize + normalize to [0,1], return:
      - original numpy RGB image
      - batch of shape (1, H, W, 3)
    """
    img = pil_img.convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, img_size)
    img_resized = img_resized.astype("float32") / 255.0
    batch = np.expand_dims(img_resized, axis=0)
    return img_np, batch

# ==========================
# 4. FHD ensemble
# ==========================

def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd(img_batch):
    """
    FHD-HybridNet:
      - Get predictions from DenseNet, MobileNet, ResNet
      - Compute average FHD for each model vs others
      - Select model with minimum average FHD
    """
    models = {
        key: load_tf_model(key)
        for key in ["DenseNet", "MobileNet", "ResNet"]
    }

    preds_list = []
    keys = ["DenseNet", "MobileNet", "ResNet"]
    for k in keys:
        preds = models[k].predict(img_batch, verbose=0)[0]
        preds_list.append(preds)

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
    return pred_idx, chosen_probs, chosen_key, models[chosen_key]

# ==========================
# 5. Grad-CAM utilities
# ==========================

def get_last_conv_layer_name(model):
    """
    Return the name of the last layer whose output is 4D (batch, H, W, C).
    Works even with Keras 3 / TF 2.20 models.
    """
    # Walk from the end of the layer list
    for layer in reversed(model.layers):
        # কিছু layer এর output_shape থাকে না, তাই try/except
        try:
            out_shape = layer.output_shape
        except Exception:
            continue

        # Conv / feature-map type layer হলে output 4-D হয়
        if isinstance(out_shape, (tuple, list)) and len(out_shape) == 4:
            return layer.name

    # যদি কিছুই না পাওয়া যায়, পরিষ্কার error দাও
    raise ValueError("No suitable conv layer found in model.")



def make_gradcam_heatmap(model, img_array, class_index=None):
    last_conv_name = get_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]   # (H, W, C)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def overlay_gradcam(heatmap, orig_image_np, alpha=0.4):
    h, w, _ = orig_image_np.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = alpha * (heatmap_color.astype("float32") / 255.0) + \
        (1 - alpha) * (orig_image_np.astype("float32") / 255.0)
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay

# ==========================
# 6. Sample gallery helper
# ==========================

def list_sample_images():
    folder = "sample_images"
    if not os.path.isdir(folder):
        return []
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort()
    return [os.path.join(folder, f) for f in files]

# ==========================
# 7. Streamlit UI
# ==========================

st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI Demo",
    layout="wide",
)

st.title("🧠 FHD-HybridNet – Alzheimer MRI Classification Demo")

st.write(
    "This app uses three CNN backbones (DenseNet121, MobileNetV1, ResNet50V2) "
    "and a fuzzy-logic-based ensemble (Fuzzy Hellinger Distance) to classify "
    "Alzheimer MRI scans into four classes."
)

# ----- Sidebar -----
with st.sidebar:
    st.header("Controls")

    model_choice = st.selectbox(
        "Select model",
        ["FHD-HybridNet", "DenseNet", "MobileNet", "ResNet"],
        index=0,
    )

    input_mode = st.radio(
        "Choose image source",
        ["Upload MRI", "Sample gallery"],
        index=0,
    )

    uploaded_file = None
    selected_sample_path = None

    if input_mode == "Upload MRI":
        uploaded_file = st.file_uploader(
            "Upload an MRI image",
            type=["png", "jpg", "jpeg"],
        )
    else:
        sample_paths = list_sample_images()
        if len(sample_paths) == 0:
            st.info("No images found in `sample_images/` folder.")
        else:
            sample_labels = [os.path.basename(p) for p in sample_paths]
            chosen = st.selectbox("Pick a sample image", sample_labels)
            if chosen:
                idx = sample_labels.index(chosen)
                selected_sample_path = sample_paths[idx]
# ------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------
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
    # sample_images ফোল্ডার থেকে ফাইল লিস্ট
    SAMPLE_DIR = "sample_images"
    sample_files = sorted(
        [f for f in os.listdir(SAMPLE_DIR)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    selected_file = st.sidebar.selectbox(
        "Pick a sample image", sample_files
    )

# 👉 নতুন Predict button
run_button = st.sidebar.button("🔍 Predict")


# ----- Main logic -----

import io  # file top এ থাকলে ভাল

st.title("FHD-HybridNet Alzheimer MRI Classifier")

st.write(
    "This app uses three CNN backbones (DenseNet121, MobileNetV1, ResNet50V2) "
    "and a fuzzy-logic-based ensemble (Fuzzy Hellinger Distance) to classify "
    "Alzheimer MRI scans into four classes."
)

# এখানে prediction চালাবো শুধু button চাপলে
if run_button:
    # ---------- 1. Image source resolve ----------
    if source == "Upload MRI":
        if uploaded_file is None:
            st.error("Please upload an MRI image first.")
            st.stop()
        # uploaded file থেকে PIL image
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        img_batch = preprocess_image_array(img_np)  # ← তোমার নিজস্ব preprocess func
        img_path_str = uploaded_file.name
    else:
        if not selected_file:
            st.error("No sample image found.")
            st.stop()
        img_path = os.path.join(SAMPLE_DIR, selected_file)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_batch = preprocess_image_array(img_np)
        img_path_str = img_path

    # ---------- 2. Run model ----------
    with st.spinner("Running prediction..."):
        if model_name == "FHD-HybridNet":
            probs, pred_idx, grad_model_name, grad_model = run_fhd_ensemble(img_batch)
            title_cam = f"FHD-HybridNet ({grad_model_name})"
        else:
            model = load_single_model(model_name)      # তোমার cache করা loader
            preds = model.predict(img_batch, verbose=0)[0]
            probs = preds
            pred_idx = int(np.argmax(preds))
            grad_model = model
            title_cam = model_name

        pred_class = CLASS_NAMES[pred_idx]  # list of labels

        # ---------- 3. Grad-CAM ----------
        heatmap = make_gradcam_heatmap(grad_model, img_batch, class_index=pred_idx)
        overlay = overlay_gradcam_on_image(heatmap, img)  # returns numpy / float image

    # ---------- 4. Figure draw ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Original
    axes[0].imshow(img)
    axes[0].set_title(f"Original\n({os.path.basename(img_path_str)})")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM\n{title_cam}")
    axes[1].axis("off")

    # Probabilities bar plot
    y_pos = np.arange(len(CLASS_NAMES))
    axes[2].barh(y_pos, probs)
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

    st.markdown(f"**Predicted class:** `{pred_class}`")

    # ---------- 5. Save result button ----------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)

    st.download_button(
        label="💾 Save result as image",
        data=buf,
        file_name="fhd_prediction_result.png",
        mime="image/png",
    )


image_available = False
pil_img = None

if (input_mode == "Upload MRI") and (uploaded_file is not None):
    pil_img = Image.open(uploaded_file)
    image_available = True
elif (input_mode == "Sample gallery") and (selected_sample_path is not None):
    pil_img = Image.open(selected_sample_path)
    image_available = True

if not image_available:
    st.info("Please upload an MRI image or select one from the sample gallery.")
    st.stop()

# Preprocess
orig_np, batch = preprocess_image(pil_img, IMG_SIZE)

# Predict
if model_choice == "FHD-HybridNet":
    pred_idx, probs, used_key, grad_model = ensemble_predict_fhd(batch)
    cam_model_name = f"FHD-HybridNet (selected: {used_key})"
else:
    model = load_tf_model(model_choice)
    preds = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    probs = preds
    grad_model = model
    cam_model_name = model_choice

pred_class = CLASS_NAMES[pred_idx]

# Grad-CAM
heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
overlay = overlay_gradcam(heatmap, orig_np)

# ----- Layout -----
col1, col2, col3 = st.columns([1, 1, 1.2])

with col1:
    st.subheader("Original")
    st.image(orig_np, use_column_width=True,
             caption=f"Predicted: {pred_class}")

with col2:
    st.subheader("Grad-CAM")
    st.image(overlay, use_column_width=True,
             caption=cam_model_name)

with col3:
    st.subheader("Class probabilities")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.barh(CLASS_NAMES, probs)
    ax.set_xlim(0, 1)
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f"{p:.3f}", va="center")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Predicted class")
    st.write(f"**{pred_class}**")

    st.markdown("### Raw probabilities")
    for cls, p in zip(CLASS_NAMES, probs):
        st.write(f"- **{cls:>18}** : {p:.4f}")
