import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ==========================
# Basic config
# ==========================
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
IMG_SIZE = (224, 224)

MODEL_PATHS = {
    "DenseNet": "models/densenet_alz_fhd.keras",
    "MobileNet": "models/mobilenet_alz_fhd.keras",
    "ResNet": "models/resnet_alz_fhd.keras",
}

# ==========================
# Helpers
# ==========================

@st.cache_resource
def load_tf_model(path: str):
    return tf.keras.models.load_model(path)


def preprocess_image(pil_img, img_size=IMG_SIZE):
    """Resize + normalize to [0,1], shape (1,H,W,3)."""
    img = pil_img.convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, img_size)
    img_resized = img_resized.astype("float32") / 255.0
    batch = np.expand_dims(img_resized, axis=0)
    return img_np, batch


def fuzzy_hellinger_distance(p1, p2):
    return 0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)


def ensemble_predict_fhd(img_batch):
    """FHD-HybridNet: pick model with minimum average FHD to others."""
    models = {
        k: load_tf_model(v) for k, v in MODEL_PATHS.items()
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


def get_last_conv_layer_name(model):
    """Find the last 4D conv-like layer for Grad-CAM."""
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            continue
        if len(out_shape) == 4:
            return layer.name
    raise ValueError("No 4D conv layer found in model.")


def make_gradcam_heatmap(model, img_array, class_index=None):
    last_conv_name = get_last_conv_layer_name(model)
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
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
# Streamlit UI
# ==========================

st.set_page_config(
    page_title="FHD-HybridNet Alzheimer MRI Demo",
    layout="wide"
)

st.title("🧠 FHD-HybridNet – Alzheimer MRI Classification Demo")

st.write(
    "Upload a brain MRI slice, choose a model, and see the predicted dementia class "
    "along with Grad-CAM heatmap and class probabilities."
)

model_choice = st.selectbox(
    "Select model",
    ["FHD-HybridNet", "DenseNet", "MobileNet", "ResNet"],
    index=0,
)

uploaded = st.file_uploader(
    "Upload an MRI image",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded)
    orig_np, batch = preprocess_image(pil_img, IMG_SIZE)

    if model_choice == "FHD-HybridNet":
        pred_idx, probs, used_key, grad_model = ensemble_predict_fhd(batch)
        cam_model_name = f"FHD-HybridNet ({used_key})"
    else:
        model = load_tf_model(MODEL_PATHS[model_choice])
        preds = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        probs = preds
        grad_model = model
        cam_model_name = model_choice

    pred_class = CLASS_NAMES[pred_idx]

    # Grad-CAM
    heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
    overlay = overlay_gradcam(heatmap, orig_np)

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

else:
    st.info("Upload an MRI image to start.")
