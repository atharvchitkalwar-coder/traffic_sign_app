import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# PAGE CONFIG
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
body {
    background-color: #0d1117;
}
.block-container {
    padding-top: 6rem !important; /* FIXED TOP SPACING 🚀 */
}
.title {
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff !important;
    text-align: center;
    text-shadow: 0px 0px 12px rgba(255,255,255,0.35);
    margin-top: 10px !important;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 1.2rem;
    color: #c9c9c9 !important;
    text-align: center;
    margin-bottom: 35px;
}
.result-box {
    background: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border-left: 5px solid #00c853;
}
.upload-note {
    background: #3b3f04;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    color: #e5e5e5;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# CLASS LABELS
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing',
    10: 'No passing >3.5t', 11: 'Right of way',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
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

# SIDEBAR INFO
st.sidebar.title("⚙️ Model Details")
st.sidebar.markdown("#### CNN Model for Traffic Sign Recognition")
st.sidebar.markdown("""
- 📁 Dataset: German Traffic Signs  
- 🧠 Classes: **43**  
- 🔹 Input: **32×32 RGB**  
- 🧮 Normalization applied  
""")

# MAIN HEADINGS
st.markdown('<p class="title">🚦 Traffic Sign Recognition System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered real-time road sign classification</p>', unsafe_allow_html=True)

# UPLOAD
uploaded_file = st.file_uploader("📌 Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

# PREDICTION
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", width=350)

    with st.spinner("🔍 Detecting sign..."):
        img = np.expand_dims(np.array(image.resize((32,32)).convert("RGB"))/255.0, axis=0)
        predictions = model.predict(img)
        top_5 = predictions[0].argsort()[-5:][::-1]
        pred_idx = top_5[0]

    st.markdown(
        f"<div class='result-box'><h4>Prediction: {class_names[pred_idx]}</h4>"
        f"<p>Confidence: <b>{predictions[0][pred_idx]*100:.2f}%</b></p></div>",
        unsafe_allow_html=True
    )

    st.bar_chart({"Confidence": [float(predictions[0][i]) for i in top_5]},
                 x=[class_names[i] for i in top_5])

else:
    st.markdown('<p class="upload-note">⬆ Upload an image to start classification</p>',
                unsafe_allow_html=True)

# FOOTER
st.write("---")
st.markdown(
    "<center>🔬 Powered by <b>TensorFlow</b> + <b>Streamlit</b> 🚀 | Developed by Atharv 👨‍💻</center>",
    unsafe_allow_html=True
)
