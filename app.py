import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "plant_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { color: #2e7d32; }
    div.stButton > button {
        background-color: #2e7d32; color: white; border: none;
        border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 600;
    }
    div.stButton > button:hover { background-color: #1b5e20; color: white; }
    .result-box {
        background-color: #f1f8e9; border-left: 6px solid #2e7d32;
        padding: 1rem 1.25rem; border-radius: 8px; margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    with open(KB_PATH, encoding="utf-8") as f:
        knowledge_base = json.load(f)
    return model, class_names, knowledge_base


def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype="float32")  # raw 0-255 values, same as training
    return np.expand_dims(array, axis=0)


st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a clear photo of a single plant leaf to identify possible diseases.")

if not (MODEL_PATH.exists() and CLASSES_PATH.exists()):
    st.error(
        "Trained model not found. Run `python training\\train_model.py` to train it, "
        "or place `plant_model.keras` and `class_names.json` inside the `models` folder."
    )
    st.stop()

model, class_names, knowledge_base = load_resources()

uploaded = st.file_uploader(
    "Upload a leaf image (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect Disease"):
        with st.spinner("Analyzing the leaf..."):
            probs = model.predict(preprocess(image), verbose=0)[0]
        idx = int(np.argmax(probs))
        label = class_names[idx]
        confidence = float(probs[idx]) * 100
        info = knowledge_base.get(label)

        title = info["name"] if info else label
        st.markdown(
            f'<div class="result-box"><h3>{title}</h3>'
            f"<b>Confidence:</b> {confidence:.1f}%</div>",
            unsafe_allow_html=True,
        )

        if confidence < 60:
            st.warning(
                "Low confidence. Try a clearer, well-lit photo of a single leaf "
                "on a plain background."
            )

        if info:
            st.subheader("About this condition")
            st.write(info["description"])
            st.subheader("Common symptoms")
            for item in info["symptoms"]:
                st.markdown(f"- {item}")
            st.subheader("Recommended treatment")
            for item in info["treatment"]:
                st.markdown(f"- {item}")
            st.subheader("Prevention tips")
            for item in info["prevention"]:
                st.markdown(f"- {item}")
        else:
            st.info("No extra information is available for this class yet.")
else:
    st.info("Please upload a leaf image to begin.")