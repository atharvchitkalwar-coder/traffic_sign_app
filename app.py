# app.py — Road Edge Theme (Final - Local Repo Image Usage)
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Traffic Sign Recognition — Road Edge Theme",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# CSS Styles + Local Image
# -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700;400&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(255,255,255,0.01), transparent 10%),
            linear-gradient(180deg, #0b0d0f 0%, #0e1113 100%);
        color: #e6eef8;
        font-family: "Roboto Condensed", sans-serif;
    }

    .hero-container {
        position: relative;
        width: 100%;
        max-width: 1200px;
        margin: 20px auto;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(2,6,10,0.7);
        min-height: 380px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
    }

    .hero-background-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('traffic_sign.png'); /* LOCAL IMAGE */
        background-size: cover;
        background-position: center bottom;
        filter: brightness(0.68);
        z-index: 1;
    }

    .hero-overlay-title {
        position: relative;
        z-index: 2;
        color: #fff;
        font-size: 52px;
        font-weight: 700;
        margin-left: 5%;
        margin-top: 50px;
        width: 45%;
        text-align: left;
        text-shadow: 0 0 20px rgba(0,0,0,0.8);
    }

    .hero-description {
        position: relative;
        z-index: 2;
        margin-top: 6px;
        color: #bfc8d9;
        font-size: 18px;
        margin-left: 5%;
        width: 45%;
        text-align: left;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 30px rgba(1,6,10,0.6);
        border: 1px solid rgba(255,255,255,0.04);
    }

    .result {
        background: rgba(0,0,0,0.4);
        border-left: 6px solid #f2c94c;
        padding: 12px;
        border-radius: 10px;
        color: white;
    }

    .footer {
        text-align: center;
        color: #9fb0c9;
        padding-top: 18px;
        padding-bottom: 30px;
        font-size: 14px;
    }

    @media (max-width: 900px) {
        .hero-overlay-title { font-size: 36px; width: 90%; text-align:center; }
        .hero-description { width: 90%; text-align:center; }
        .hero-container { min-height: 250px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_tf_model():
    return tf.keras.models.load_model("model.h5")

model = load_tf_model()

# -------------------------
# Classes
# -------------------------
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing', 10: 'No passing >3.5t',
    11: 'Right of way', 12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: '>3.5t prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Left curve', 20: 'Right curve', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows (right)',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Ice/snow',
    31: 'Animals crossing', 32: 'End restrictions', 33: 'Turn right',
    34: 'Turn left', 35: 'Ahead only', 36: 'Straight or right',
    37: 'Straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout', 41: 'End no passing', 42: 'End no passing >3.5t'
}

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image("traffic_sign.png", width=120)  # LOCAL IMAGE APPEARING IN SIDEBAR
st.sidebar.title("Model Info")
st.sidebar.write("• CNN Model\n• 43 Classes\n• GTSRB Dataset\n• Input: 32×32 RGB")

# -------------------------
# Hero Section
# -------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-background-image"></div>
        <h1 class="hero-overlay-title">Traffic Sign Detection</h1>
        <p class="hero-description">AI-powered real-time recognition 🚦</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Main Columns
# -------------------------
col1, col2 = st.columns([1,1.3])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("📤 Upload Traffic Sign")
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("📊 Prediction")
    result_box = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Prediction Logic
# -------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    x = img.resize((32,32))
    x = np.array(x, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    with st.spinner("Processing..."):
        pred = model.predict(x)[0]
        idx = np.argmax(pred)
        conf = float(pred[idx]) * 100

    result_box.success(f"✔ {class_names[idx]} — {conf:.2f}% confidence")

else:
    result_box.info("Upload an image to begin prediction.")

# -------------------------
# Footer
# -------------------------
st.markdown("<div class='footer'>🚦 AI Traffic Sign Recognition — Developed by Atharv</div>", unsafe_allow_html=True)
