import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS for better UI
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
    color: #ffffff;
    text-align: center;
}
.subtitle {
    font-size: 1.2rem;
    color: #cccccc;
    text-align: center;
    margin-bottom: 20px;
}
.result-box {
    padding: 20px;
    background-color: #161b22;
    color: white;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
    border: 2px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# Load Model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Class Labels
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

# Sidebar Information
st.sidebar.title("⚙️ Model Details")
st.sidebar.write("""
**CNN Model**
- Dataset: German Traffic Signs  
- Input Size: 32×32  
- Classes: 43  
""")
st.sidebar.info("Upload a traffic sign & get real-time classification.")

# Main Title
st.markdown('<p class="title">🚦 Traffic Sign Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Using Deep Learning for Intelligent Road Safety</p>', unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("📌 Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing Image..."):
            # Preprocess
            img = image.resize((32, 32))
            img = img.convert("RGB")
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            predictions = model.predict(img)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100

        # Show Results
        st.markdown(f"""
        <div class="result-box">
            <h3>Prediction: {class_names[class_index]}</h3>
            <h4>Confidence: {confidence:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Chart
        st.write("📊 Class Probability Distribution")
        st.bar_chart(predictions[0])
else:
    st.warning("⬆️ Upload an image to begin classification")

# Footer
st.markdown("""
<hr>
<div style='text-align:center; color: #777;'>

Using TensorFlow + Streamlit 🚀
</div>
""", unsafe_allow_html=True)

