import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown

from itertools import combinations
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D

# ---------------------------------------------------------------------
# 0. BASIC SETUP
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="FCI-ResNet Alzheimer MRI",
    layout="wide"
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_CACHE_DIR = "models_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Google Drive IDs (.h5)
H5_RESNET50_ID = "1H7iUcI94ul01BWqtGwf-LZ2jKnATf17l"
H5_RESNET101_ID = "11UB7-lHDqcAuPNnA_q8wAQrgS2CijHV6"
H5_RESNET152_ID = "1Rg0d2efE-oV_usoJkr6F2xXLCRW9yPTi"

MODEL_FILES = {
    "ResNet50": ("az_model_resnet50.h5", H5_RESNET50_ID),
    "ResNet101": ("az_model_resnet101.h5", H5_RESNET101_ID),
    "ResNet152": ("az_model_resnet152.h5", H5_RESNET152_ID),
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
            "2) Verify FILE IDs are correct\n"
            f"Bad file: {path} ({size} bytes)"
        )

def _download_if_needed(file_id: str, filename: str) -> str:
    if not file_id or "PASTE_" in file_id:
        raise RuntimeError(
            "Model IDs are not set.\n"
            "Open app.py and set correct Google Drive FILE IDs."
        )

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

# ---------------------------------------------------------------------
# 1) BUILD MODELS (matching training architecture)
# ---------------------------------------------------------------------
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
    if model_name == "ResNet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)
    if model_name == "ResNet101":
        base = tf.keras.applications.ResNet101(include_top=False, weights=None, input_shape=(224, 224, 3))
        return _build_head(base)
    if model_name == "ResNet152":
        base = tf.keras.applications.ResNet152(include_top=False, weights=None, input_shape=(224, 224, 3))
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
def load_models_batch(selected_names: tuple):
    """Load multiple models once."""
    loaded = {}
    for name in selected_names:
        loaded[name] = load_single_model(name)
    return loaded

# ---------------------------------------------------------------------
# 2) IMAGE HELPERS
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
# 3. GRAD-CAM HELPERS
# ---------------------------------------------------------------------
def get_last_conv_layer_name(model):
    """Try to find a suitable last convolution layer for Grad-CAM."""
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
    """Robust Grad-CAM."""
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
# 4) FCI (CHOQUET INTEGRAL) FUSION HELPERS (Sugeno λ-measure)
# ---------------------------------------------------------------------
def normalize_weights(w):
    w = np.array(w, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    return (w / np.sum(w)).tolist()

def solve_sugeno_lambda(singletons, max_iter=200, tol=1e-10):
    """Solve λ from: Π(1 + λ g_i) = 1 + λ"""
    g = np.array(singletons, dtype=np.float64)
    eps = 1e-9

    def f(lam):
        return np.prod(1.0 + lam * g) - (1.0 + lam)

    if abs(np.sum(g) - 1.0) < 1e-8:
        return 0.0

    lo = -1.0 + eps
    hi = 1.0
    flo = f(lo)

    for _ in range(60):
        fhi = f(hi)
        if flo * fhi < 0:
            break
        hi *= 2.0
        if hi > 1e6:
            return 0.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid < 0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return (lo + hi) / 2.0

def sugeno_capacity(subset_idx, singletons, lam):
    """g(S) = (Π(1 + λ g_i) - 1) / λ"""
    g = np.array(singletons, dtype=np.float64)
    if len(subset_idx) == 0:
        return 0.0
    if abs(lam) < 1e-12:
        return float(np.sum(g[list(subset_idx)]))
    prod_term = np.prod(1.0 + lam * g[list(subset_idx)])
    return float((prod_term - 1.0) / lam)

def choquet_integral_vector(x_vec, singletons):
    """Choquet integral for a single class."""
    x = np.array(x_vec, dtype=np.float64)
    M = x.shape[0]
    lam = solve_sugeno_lambda(singletons)

    order = np.argsort(-x)
    x_sorted = x[order]
    x_next = np.append(x_sorted[1:], 0.0)

    total = 0.0
    top_set = []
    for k in range(M):
        top_set.append(order[k])
        gAk = sugeno_capacity(top_set, singletons, lam)
        total += (x_sorted[k] - x_next[k]) * gAk
    return float(total)

def fci_fuse_probs(probs_list, weights):
    """Fuse probabilities using Choquet Integral."""
    probs = np.stack(probs_list, axis=0)  # (M, C)
    M, C = probs.shape
    fused = np.zeros((C,), dtype=np.float64)

    for c in range(C):
        fused[c] = choquet_integral_vector(probs[:, c], weights)

    fused = np.maximum(fused, 0.0)
    s = float(np.sum(fused))
    if s <= 1e-12:
        fused = np.ones((C,), dtype=np.float64) / C
    else:
        fused = fused / s
    return fused.astype(np.float32)

def run_fci_ensemble(img_batch, selected_models, weights):
    """Run FCI ensemble."""
    selected_names = list(selected_models.keys())

    probs_list = []
    for name in selected_names:
        p = selected_models[name].predict(img_batch, verbose=0)[0]
        probs_list.append(p)

    fused_probs = fci_fuse_probs(probs_list, weights)
    pred_idx = int(np.argmax(fused_probs))

    # Pick representative model for Grad-CAM
    w = np.array(weights, dtype=np.float64)
    best_w = np.max(w)
    cand = np.where(np.isclose(w, best_w))[0].tolist()
    if len(cand) == 1:
        rep_i = cand[0]
    else:
        confs = [probs_list[i][pred_idx] for i in cand]
        rep_i = cand[int(np.argmax(confs))]

    rep_name = selected_names[rep_i]
    rep_model = selected_models[rep_name]

    return fused_probs, pred_idx, rep_name, rep_model

# ---------------------------------------------------------------------
# 5) UI HELPERS: BUILD COMBINATIONS LIST
# ---------------------------------------------------------------------
def build_combo_label(combo):
    names_txt = " , ".join(combo)
    return f"{len(combo)}-models | {names_txt}"

def all_combinations(names):
    combos = []
    N = len(names)
    for r in range(2, N + 1):
        for comb in combinations(names, r):
            combos.append(comb)
    return combos

ALL_MODEL_NAMES = list(MODEL_FILES.keys())
ALL_COMBOS = all_combinations(ALL_MODEL_NAMES)
ALL_COMBO_LABELS = [build_combo_label(c) for c in ALL_COMBOS]
LABEL_TO_COMBO = {lab: comb for lab, comb in zip(ALL_COMBO_LABELS, ALL_COMBOS)}

# ---------------------------------------------------------------------
# 6. SIDEBAR CONTROLS
# ---------------------------------------------------------------------
st.sidebar.title("Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Single Model", "FCI-ResNet (Choquet Fusion)"],
    index=1
)

source = st.sidebar.radio(
    "Choose image source",
    ["Upload MRI", "Sample gallery"],
    index=0
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
    SAMPLE_DIR = "sample_images"
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

st.sidebar.markdown("---")

# Model selection UI
selected_single = None
selected_combo = None
selected_custom = None
weights = None

if mode == "Single Model":
    selected_single = st.sidebar.selectbox(
        "Select a model",
        ALL_MODEL_NAMES,
        index=0
    )

else:
    st.sidebar.caption("Choose models for FCI-ResNet")

    combo_mode = st.sidebar.radio(
        "Combination selection",
        ["Auto (pick from all combinations)", "Custom (manual select)"],
        index=0
    )

    if combo_mode == "Auto (pick from all combinations)":
        size_filter = st.sidebar.multiselect(
            "Combo size",
            options=list(range(2, len(ALL_MODEL_NAMES) + 1)),
            default=[3]
        )

        filtered_labels = [
            lab for lab, combo in zip(ALL_COMBO_LABELS, ALL_COMBOS)
            if not size_filter or len(combo) in size_filter
        ]

        if not filtered_labels:
            st.sidebar.error("No combinations match your filters.")
            st.stop()

        selected_label = st.sidebar.selectbox("Select a combination", filtered_labels, index=0)
        selected_combo = LABEL_TO_COMBO[selected_label]

    else:
        selected_custom = st.sidebar.multiselect(
            "Pick models (min 2)",
            options=ALL_MODEL_NAMES,
            default=[ALL_MODEL_NAMES[0], ALL_MODEL_NAMES[1]]
        )
        if selected_custom is None or len(selected_custom) < 2:
            st.sidebar.warning("Select at least 2 models for FCI-ResNet.")
            selected_custom = None
        else:
            selected_combo = tuple(selected_custom)

    # Weights sliders
    if selected_combo is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Singleton weights (importance)")
        st.sidebar.caption("Normalized automatically.")

        raw_w = []
        for name in selected_combo:
            raw = st.sidebar.slider(
                f"{name}",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05
            )
            raw_w.append(raw)

        if float(np.sum(raw_w)) <= 1e-9:
            st.sidebar.error("At least one weight must be > 0.")
            st.stop()
        weights = normalize_weights(raw_w)

run_button = st.sidebar.button("▶ Run prediction")

# ---------------------------------------------------------------------
# 7. MAIN AREA HEADER
# ---------------------------------------------------------------------
st.title("FCI-ResNet Alzheimer MRI Detection")

st.markdown(
    """
This app performs:
- **Single model inference** with ResNet50, ResNet101, or ResNet152, or  
- **FCI-ResNet** using **Choquet Integral (Sugeno λ-measure)** decision fusion across selected models.

Models are downloaded from Google Drive on first use.
"""
)

col_info, col_out = st.columns([1, 1])

with col_info:
    st.info("👈 Select mode, models & image, then click **Run prediction**.")
    st.subheader("Selected options")
    st.write(f"**Mode:** {mode}")
    st.write(f"**Source:** {source}")

    if mode == "Single Model":
        st.write(f"**Model:** {selected_single}")
    else:
        st.write(f"**FCI Models ({len(selected_combo) if selected_combo else 0}):**")
        if selected_combo:
            for i, n in enumerate(selected_combo):
                st.write(f"- {n} | weight={weights[i]:.3f}")
        else:
            st.write("_No combination selected yet._")

    if isinstance(chosen_file, str):
        st.write(f"**Image path:** `{chosen_file}`")
    elif chosen_file is None:
        st.write("_No image selected yet._")
    else:
        st.write(f"**Uploaded file:** `{chosen_file.name}`")

if not run_button:
    st.stop()

if chosen_file is None:
    st.error("Please upload or select an image first.")
    st.stop()

# ---------------------------------------------------------------------
# 8. LOAD IMAGE & RUN MODEL
# ---------------------------------------------------------------------
orig_img, batch = load_image_from_file(chosen_file, IMG_SIZE)

with st.spinner("Running prediction…"):
    if mode == "Single Model":
        model = load_single_model(selected_single)
        preds = model.predict(batch, verbose=0)[0]
        probs = preds.astype(np.float32)
        pred_idx = int(np.argmax(probs))
        grad_model = model
        cam_title = selected_single
        chosen_key = selected_single

    else:
        selected_names = tuple(selected_combo)
        models_dict = load_models_batch(selected_names)

        probs, pred_idx, rep_name, grad_model = run_fci_ensemble(
            img_batch=batch,
            selected_models=models_dict,
            weights=weights
        )
        cam_title = f"FCI-ResNet (Rep: {rep_name})"
        chosen_key = rep_name

pred_class = CLASS_NAMES[pred_idx]

with st.spinner("Computing Grad-CAM…"):
    heatmap = make_gradcam_heatmap(grad_model, batch, class_index=pred_idx)
    overlay = overlay_heatmap_on_image(heatmap, orig_img, alpha=0.45)

# ---------------------------------------------------------------------
# 9. PREDICTION OUTPUT
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Prediction Output")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(orig_img)
axes[0].set_title(f"Original Image\nClass Name: {pred_class}")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title(f"Grad-CAM\n{cam_title}")
axes[1].axis("off")

axes[2].barh(CLASS_NAMES, probs)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Probability")
axes[2].set_title("Class Probabilities")

for i, cls in enumerate(CLASS_NAMES):
    axes[2].text(float(probs[i]) + 0.01, i, f"{float(probs[i]):.3f}", va="center")

plt.tight_layout()
st.pyplot(fig)

st.markdown(f"#### Final Prediction : *{pred_class}*")

if mode != "Single Model":
    st.caption(f"Grad-CAM shown using representative model: **{chosen_key}** (from selected ensemble).")

# ---------------------------------------------------------------------
# 10. Save generated image
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
