import streamlit as st
import tensorflow as tf
import os

MODEL_PATH = "models/final_trained_model.h5"  # adjust if different

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    try:
        # Try normal load (legacy .h5)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.warning(f"Standard load failed: {e}")
        st.info("Trying fallback legacy loader…")
        try:
            # Fallback: legacy loader (works in some envs)
            from keras.saving import legacy as legacy_saving  # might not exist in older installs
            model = legacy_saving.load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e2:
            st.error(f"Failed to load model with both methods:\n{e2}")
            st.stop()

model = load_trained_model()
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

