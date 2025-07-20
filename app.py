import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Path to the trained model
MODEL_PATH = "models/final_trained_model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_trained_model()
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    img = image.resize((224, 224))  # resize for model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        preds = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds) * 100
        st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}%)")
