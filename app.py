import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import base64

# ------------------------- #
# Page Config
# ------------------------- #
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# ------------------------- #
# Load Base64 Background Image
# ------------------------- #
def get_base64_image(image_file):
    with open(image_file, "rb") as img:
        return base64.b64encode(img.read()).decode()

image_base64 = get_base64_image("traffic_sign.png")

# ------------------------- #
# CSS - Hero Background + UI Style
# ------------------------- #
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700;400&display=swap');

    .stApp {{
        background: #0f1115;
        font-family: "Roboto Condensed", sans-serif;
        color: #e6eef8;
    }}

    .hero-container {{
        position: relative;
        width: 100%;
        height: 400px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 30px;
    }}

    .hero-background-image {{
        position: absolute;
        width: 100%;
        height: 100%;
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        filter: brightness(0.45) blur(1px);
        z-index: 1;
    }}

    .hero-overlay-title {{
        position: relative;
        z-index: 2;
        font-size: 64px;
        font-weight: 900;
        color: white;
        padding-left: 50px;
        padding-top: 80px;
        text-shadow: 0 0 15px rgba(0,0,0,0.85);
    }}

    .hero-description {{
        position: relative;
        z-index: 2;
        color: #d9dce2;
        font-size: 20px;
        padding-left: 50px;
        text-shadow: 0 0 10px rgba(0,0,0,0.8);
    }}

    .card {{
        background: rgba(255,255,255,0.03);
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.05);
    }}

    .result {{
        background: rgba(242,201,76,0.08);
        border-left: 6px solid #f2c94c;
        padding: 14px;
        border-radius: 8px;
        color: #fff;
    }}

    .footer {{
        text-align:center;
        color:#9fb0c9;
        padding: 25px;
        font-size:14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- #
# Load TensorFlow Model
# ------------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ------------------------- #
# Class Labels
# ------------------------- #
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing >3.5t', 11: 'Right of way', 12: 'Priority road',
    13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: '>3.5t prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Left curve', 20: 'Right curve', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows (right)', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Ice/snow', 31: 'Animals crossing',
    32: 'End restrictions', 33: 'Turn right', 34: 'Turn left', 35: 'Ahead only',
    36: 'Straight or right', 37: 'Straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout', 41: 'End no passing', 42: 'End no passing >3.5t'
}

# ------------------------- #
# HERO SECTION
# ------------------------- #
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-background-image"></div>
        <div class="hero-overlay-title">Traffic Sign Detection</div>
        <p class="hero-description">AI-powered classification for road safety 🚦</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------- #
# Layout
# ------------------------- #
left, right = st.columns([1, 1.3])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📤 Upload Traffic Sign")
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    result_placeholder = st.empty()

# ------------------------- #
# Prediction Section
# ------------------------- #
if file:
    img = Image.open(file).convert("RGB")

    with left:
        st.image(img, caption="Uploaded Image Preview", use_column_width=True)

    arr = np.array(img.resize((32, 32)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("🔍 Detecting traffic sign..."):
        preds = model.predict(arr)[0]
        top5 = preds.argsort()[-5:][::-1]

    main_idx = top5[0]
    confidence = preds[main_idx] * 100

    with right:
        result_placeholder.markdown(
            f"""
            <div class='result'>
                <h3>Prediction: {class_names[int(main_idx)]}</h3>
                <p>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    with right:
        st.info("Upload a traffic sign image for prediction 👆")

# ------------------------- #
# Footer
# ------------------------- #
st.markdown("<div class='footer'>🚀 Powered by TensorFlow & Streamlit | Developed by Atharv ❤️</div>", unsafe_allow_html=True)
