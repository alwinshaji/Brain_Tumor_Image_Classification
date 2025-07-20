import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ----------------- CONFIG -----------------
MODEL_PATH = "final_trained_model.h5"
DRIVE_FILE_ID = "1otdiwo82KkKWNMebhTI7BUDQf55-mB4H"  # Your Google Drive file ID
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMG_SIZE = (224, 224)

# ----------------- MODEL LOADER -----------------
@st.cache_resource
def load_trained_model():
    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Download model if missing
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive... This may take a few minutes.")
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_trained_model()

# ----------------- IMAGE PREPROCESS -----------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dim
    return image

# ----------------- PREDICTION -----------------
def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return pred_class, confidence, preds[0]

# ----------------- STREAMLIT UI -----------------
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI brain image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        pred_class, confidence, all_preds = predict(image)
        st.success(f"**Prediction:** {pred_class} ({confidence:.2f}% confidence)")

        # Show probabilities for each class
        st.subheader("Class Probabilities:")
        for class_name, prob in zip(CLASS_NAMES, all_preds):
            st.write(f"{class_name}: {prob * 100:.2f}%")
