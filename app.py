import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ───────────────────────────────────────── UI INITIAL SETUP ───────────────────────────────────────── #
st.set_page_config( 
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>
body {
    background-color: #0d1117;
}
.block-container {
    padding-top: 2rem;
}
.title {
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff !important;
    text-align: center;
    text-shadow: 0px 0px 12px rgba(255,255,255,0.35);
    margin-top: 25px;   /* 🚀 Added Space */
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


# ───────────────────────────────────────── LOAD MODEL ───────────────────────────────────────── #
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Class Names
class_names = {
      0: 'Speed limit (20km/h)', 
    1: 'Speed limit (30km/h)', 
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 
    4: 'Speed limit (70km/h)', 
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 
    8: 'Speed limit (120km/h)',
    9: 'No passing', 
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 
    12: 'Priority road', 
    13: 'Yield',
    14: 'Stop', 
    15: 'No vehicles', 
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 
    18: 'General caution', 
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 
    21: 'Double curve', 
    22: 'Bumpy road',
    23: 'Slippery road', 
    24: 'Road narrows on the right', 
    25: 'Road work',
    26: 'Traffic signals', 
    27: 'Pedestrians', 
    28: 'Children crossing',
    29: 'Bicycles crossing', 
    30: 'Beware of ice/snow', 
    31: 'Wild animals crossing',
    32: 'End of all speed/no-passing zones', 
    33: 'Turn right ahead', 
    34: 'Turn left ahead', 
    35: 'Ahead only', 
    36: 'Go straight or right',
    37: 'Go straight or left', 
    38: 'Keep right', 
    39: 'Keep left',
    40: 'Roundabout mandatory', 
    41: 'End of no passing',
    42: 'End of no passing for vehicles >3.5 metric tons'
}

# ───────────────────────────────────────── SIDEBAR ───────────────────────────────────────── #
st.sidebar.title("⚙️ Model Details")
st.sidebar.markdown("#### CNN Model for Traffic Sign Classification")
st.sidebar.markdown("""
-  Dataset: German Traffic Signs  
-  Classes: **43**  
-  Input: **32×32 RGB**  
-  Normalization applied  
""")
st.sidebar.info("Upload a traffic sign image to classify it!")

# ───────────────────────────────────────── MAIN TITLE ───────────────────────────────────────── #
st.markdown('<p class="title">🚦 Traffic Sign Recognition System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered real-time road sign classification</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📌 Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

# ───────────────────────────────────────── PREDICTION HANDLING ───────────────────────────────────────── #
if uploaded_file:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", width=350)

    with st.spinner("🔍 Analyzing image..."):
        img = image.resize((32, 32)).convert("RGB")
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        top_5_idx = predictions[0].argsort()[-5:][::-1]

        # Display Result
        main_pred = top_5_idx[0]
        confidence = predictions[0][main_pred]

    st.markdown(
        f"<div class='result-box'><h4>Prediction: {class_names[main_pred]}</h4>"
        f"<p>Model Confidence: <b>{confidence*100:.2f}%</b></p></div>",
        unsafe_allow_html=True
    )

    # Display chart for top-5
    top5_labels = [class_names[i] for i in top_5_idx]
    top5_scores = [float(predictions[0][i]) for i in top_5_idx]
    st.bar_chart({"Confidence": top5_scores}, x=top5_labels)

else:
    st.markdown('<p class="upload-note">⬆ Upload an image to begin classification</p>',
                unsafe_allow_html=True)

# ───────────────────────────────────────── FOOTER ───────────────────────────────────────── #
st.write("---")
st.markdown(
    "<center>🔬 Powered by <b>TensorFlow</b> + <b>Streamlit</b> 🚀 | ",
    unsafe_allow_html=True
)



