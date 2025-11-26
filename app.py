import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import base64

# ------------------------- #
# PAGE CONFIG
# ------------------------- #
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------- #
# LOAD IMAGE FOR HERO BACKGROUND
# ------------------------- #
def get_base64_image(image_file):
    with open(image_file, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_base64 = get_base64_image("traffic_sign.png")


# ------------------------- #
# CSS
# ------------------------- #
st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700;400&display=swap');

    .stApp {{
        background: #0f1115;
        font-family: 'Roboto Condensed', sans-serif;
    }}

    .hero {{
        position: relative;
        width: 100%;
        height: 340px;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 25px;
    }}

    .hero-bg {{
        position: absolute;
        width: 100%;
        height: 100%;
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        filter: brightness(0.45) blur(1px);
    }}

    .hero-title {{
        position: relative;
        z-index: 2;
        padding: 70px 40px;
        font-size: 60px;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(0,0,0,0.7);
    }}

    .hero-sub {{
        position: relative;
        z-index: 2;
        font-size: 22px;
        margin-left: 40px;
        color: #e0e4eb;
    }}

    .card {{
        background: rgba(255,255,255,0.04);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 5px 18px rgba(0,0,0,0.4);
    }}

    .result-box {{
        background: rgba(255,215,0,0.08);
        padding: 12px;
        border-left: 6px solid #ffd740;
        border-radius: 8px;
        margin-bottom: 12px;
    }}

    .footer {{
        padding: 20px;
        text-align: center;
        font-size: 14px;
        color: #9fa9bd;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------- #
# LOAD MODEL
# ------------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()


# ------------------------- #
# CLASS NAMES
# ------------------------- #
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End speed limit 80', 7: 'Speed limit (100)', 8: 'Speed limit (120)',
    9: 'No passing', 10: 'No passing >3.5t', 11: 'Right of way',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: '>3.5t prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Left curve', 20: 'Right curve', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows (right)',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Ice/snow',
    31: 'Animals crossing', 32: 'End restrictions',
    33: 'Turn right', 34: 'Turn left', 35: 'Ahead only',
    36: 'Straight or right', 37: 'Straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout', 41: 'End no passing',
    42: 'End no passing >3.5t'
}



# ------------------------- #
# SIDEBAR INFO
# ------------------------- #
st.sidebar.image("traffic_sign.png", width=140)
st.sidebar.markdown("## 📊 Model Info")
st.sidebar.markdown("""
- **CNN Model**
- **43 Traffic Classes**
- **Dataset: GTSRB**
- **Input: 32×32 RGB**
- Normalized pixel values
""")
st.sidebar.caption("Upload a real traffic sign for accurate detection 🚦")


# ------------------------- #
# HERO SECTION
# ------------------------- #
st.markdown(
    """
    <div class="hero">
        <div class="hero-bg"></div>
        <h1 class="hero-title">Traffic Sign Detection</h1>
        <p class="hero-sub">Real-time recognition for road safety</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------- #
# MAIN LAYOUT
# ------------------------- #
left, right = st.columns([1, 1.4])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📥 Upload Traffic Sign")
    file = st.file_uploader("Choose image file", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)


with right:
    result_area = st.empty()


# ------------------------- #
# INFERENCE
# ------------------------- #
if file:
    img = Image.open(file).convert("RGB")

    with left:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    arr = np.array(img.resize((32, 32)), dtype=np.float32) / 255.0
    arr = arr.reshape(1, 32, 32, 3)

    with st.spinner("⚙ Running Model..."):
        preds = model.predict(arr)[0]
        top5_idx = preds.argsort()[-5:][::-1]

    main_idx = top5_idx[0]

    with right:
        result_area.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction: {class_names[main_idx]}</h3>
                <p>Confidence: <b>{preds[main_idx]*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 🔹 Top-5 Bar Chart
        st.subheader("📊 Confidence Scores (Top 5)")
        labels = [class_names[i] for i in top5_idx]
        values = [preds[i] for i in top5_idx]
        st.bar_chart(pd.DataFrame({"Confidence": values}, index=labels))

else:
    st.info("📌 Upload a traffic sign image to start prediction!")


# ------------------------- #
# FOOTER
# ------------------------- #
st.markdown("<div class='footer'>🚀 Powered by TensorFlow + Streamlit </div>", unsafe_allow_html=True)


