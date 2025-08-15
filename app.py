import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import torch

# ================== Konfigurasi Path Model (relative terhadap repo) ==================
CNN_MODEL_PATH = "models/cnn_soybean_rust.keras"
YOLO_MODEL_PATH = "models/best.pt"

# ================== Load Model (cached) ==================
@st.cache_resource(show_spinner=True)
def load_cnn(path: str):
    return tf.keras.models.load_model(path)

@st.cache_resource(show_spinner=True)
def load_yolo(path: str):
    return YOLO(path)

# ================== UI ==================
st.set_page_config(page_title="Deteksi Soybean Rust — CNN → YOLO", layout="wide")
st.title("🫘 Deteksi Penyakit Daun Kedelai — Pipeline CNN → YOLOv8")

with st.sidebar:
    st.header("⚙️ Pengaturan Inference")
    st.caption("YOLO hanya dijalankan jika CNN memprediksi *Soybean Rust*.")
    conf = st.slider("Confidence YOLO", 0.05, 0.95, 0.45, 0.05)
    iou = st.slider("IoU NMS YOLO", 0.3, 0.95, 0.65, 0.05)
    st.divider()
    st.caption("Filter luas bbox (rasio terhadap area gambar). Gunakan untuk mencegah bbox raksasa.")
    min_area = st.slider("Min area bbox", 0.0, 0.20, 0.01, 0.01)
    max_area = st.slider("Max area bbox", 0.20, 1.00, 0.40, 0.01)
    st.divider()
    st.caption("Urutan label CNN (sesuaikan dengan training).")
    classes_text = st.text_input("Label CNN (dipisah koma)", value="Daun Sehat,Soybean Rust")
    CLASS_NAMES = [c.strip() for c in classes_text.split(",") if c.strip()] or ["Daun Sehat","Soybean Rust"]
    st.divider()
    st.caption(f"📁 Path model di repo:\n• CNN: {CNN_MODEL_PATH}\n• YOLO: {YOLO_MODEL_PATH}")

# Cek ketersediaan file model
cnn_ready = pathlib.Path(CNN_MODEL_PATH).exists()
yolo_ready = pathlib.Path(YOLO_MODEL_PATH).exists()

col_info = st.columns(2)
with col_info[0]:
    st.markdown("**Status Model CNN**")
    st.write("✅ Ditemukan" if cnn_ready else "❌ Tidak ditemukan")
with col_info[1]:
    st.markdown("**Status Model YOLO**")
    st.write("✅ Ditemukan" if yolo_ready else "❌ Tidak ditemukan")

if not cnn_ready:
    st.error(f"File model CNN tidak ditemukan di: `{CNN_MODEL_PATH}`. Pastikan file ada di repo.")
if not yolo_ready:
    st.warning(f"File model YOLO tidak ditemukan di: `{YOLO_MODEL_PATH}`. YOLO hanya akan berjalan jika file tersedia.")

# Muat model jika ada
cnn_model = load_cnn(CNN_MODEL_PATH) if cnn_ready else None
yolo_model = load_yolo(YOLO_MODEL_PATH) if yolo_ready else None

# ================== Upload Gambar ==================
uploaded = st.file_uploader("Unggah gambar daun kedelai", type=["jpg","jpeg","png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # ===== Tahap 1: CNN (Klasifikasi Sehat vs Rust) =====
    if cnn_model is None:
        st.error("Model CNN belum tersedia, tidak bisa melakukan klasifikasi.")
        st.stop()

    # Asumsi input 224x224 — sesuaikan jika berbeda saat training
    target_size = (224, 224)
    arr = np.array(image.resize(target_size), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Inferensi CNN..."):
        preds = cnn_model.predict(arr)
        idx = int(np.argmax(preds, axis=1)[0])
        conf_cnn = float(preds[0][idx])
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class-{idx}"

    st.subheader("🧠 Hasil CNN")
    st.write(f"Prediksi: **{label}**")
    st.write(f"Probabilitas: **{conf_cnn*100:.2f}%**")

    # ===== Tahap 2: YOLO (Deteksi Lesi) — hanya jika Rust =====
    if "rust" in label.lower() or label.strip().lower() == "soybean rust":
        if yolo_model is None:
            st.warning("Model YOLO belum tersedia, lewati deteksi.")
        else:
            st.subheader("🎯 Hasil YOLOv8 (setelah filter area)")
            with st.spinner("Inferensi YOLOv8..."):
                results = yolo_model.predict(
                    source=np.array(image),
                    imgsz=640,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )
            res = results[0]
            im_h, im_w = res.orig_img.shape[:2]

            # Filter area bbox
            filtered = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                area_ratio = (w * h) / float(im_w * im_h)
                conf_score = float(b.conf[0])
                if (min_area <= area_ratio <= max_area) and (conf_score >= conf):
                    filtered.append(b)

            kept = len(filtered)
            total = len(res.boxes) if hasattr(res, "boxes") else 0

            if kept > 0:
                # Ganti boxes di res menjadi yang difilter
                res.boxes = type(res.boxes)(torch.stack([b.data[0] for b in filtered]))
                st.caption(f"Bbox disimpan: {kept} / {total} (min_area={min_area:.2f}, max_area={max_area:.2f}, conf≥{conf:.2f})")
                st.image(res.plot(), caption="YOLOv8 (bbox difilter)", use_column_width=True)
            else:
                st.warning(f"Tidak ada bbox yang lolos filter. Total prediksi awal: {total}. Coba turunkan min_area/naikkan max_area atau turunkan conf.")

    else:
        st.success("Daun terdeteksi **SEHAT** oleh CNN → YOLO tidak dijalankan.")
