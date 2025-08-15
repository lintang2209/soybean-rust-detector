import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

# Path model di repo
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"

# Load models
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

# Class names
CLASS_NAMES = ["Daun Sehat", "Soybean Rust"]

st.title("📷 Deteksi Penyakit Daun Kedelai (Pipeline CNN → YOLOv8)")

uploaded_file = st.file_uploader("Unggah gambar daun kedelai", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # ==== Tahap 1: Prediksi CNN ====
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    cnn_pred = cnn_model.predict(img_array)
    cnn_class_idx = np.argmax(cnn_pred)
    cnn_class = CLASS_NAMES[cnn_class_idx]
    cnn_conf = np.max(cnn_pred)

    st.subheader("Hasil Prediksi CNN")
    st.write(f"Prediksi: **{cnn_class}**")
    st.write(f"Probabilitas: {cnn_conf*100:.2f}%")

    # ==== Tahap 2: YOLO hanya jika CNN prediksi Soybean Rust ====
    if cnn_class == "Soybean Rust":
        st.subheader("Hasil Prediksi YOLOv8")
        yolo_results = yolo_model.predict(
            np.array(image),
            imgsz=6
