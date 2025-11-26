# app.py — Road Edge Theme (R6: Final Image Path Update - Root Directory)
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
# CSS / Theme (Road Image Background + Dark Theme)
# -------------------------
st.markdown(
    """
    <style>
    /* Load a condensed, bold font for headings */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700;400&display=swap');

    /* Page background: Dark asphalt texture for the main app body */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(255,255,255,0.01), transparent 10%),
            linear-gradient(180deg, #0b0d0f 0%, #0e1113 100%);
        color: #e6eef8;
        font-family: "Roboto Condensed", sans-serif;
    }

    /* Hero section with the custom background image and overlaid title */
    .hero-container {
        position: relative; /* Essential for positioning child elements */
        width: 100%;
        max-width: 1200px; /* Limit max width for better appearance */
        margin: 20px auto; /* Center the hero container */
        border-radius: 14px;
        overflow: hidden; /* Ensures image and title stay within bounds */
        box-shadow: 0 8px 40px rgba(2,6,10,0.7), inset 0 2px 0 rgba(255,255,255,0.015);
        min-height: 380px; /* Ensure sufficient height for the image */
        display: flex;
        flex-direction: column; /* Allows vertical alignment */
        align-items: flex-start; /* Align content to the left side */
        justify-content: flex-start; /* Align content to the top */
    }

    .hero-background-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        /* *** FINAL IMAGE PATH FIX: Referencing image in the root directory *** */
        /* IMPORTANT: Ensure 'traffic_sign.jpg' is directly next to 'app.py' */
        background-image: url('./traffic_sign.jpg'); 
        background-size: cover;
        background-position: center bottom; /* Keep road visible */
        filter: brightness(0.65); /* Darker dimming for better contrast */
        z-index: 1; 
    }

    /* Title styling for the overlaid text */
    .hero-overlay-title {
        position: relative;
        z-index: 2; 
        color: #fff;
        font-family: "Roboto Condensed", sans-serif;
        letter-spacing: 2px;
        font-size: 52px;
        font-weight: 700;
        text-shadow:
            0 0 10px rgba(255,255,255,0.3),
            0 0 20px rgba(255, 200, 60, 0.2),
            0 8px 30px rgba(0,0,0,0.8);
        
        /* Positioning: move into the vacant sky area */
        margin-left: 5%; /* Start slightly in from the left edge */
        margin-top: 50px; /* Push down from the top edge */
        width: 45%; /* Constrain width to the left vacant space */
        text-align: left;
    }

    /* Description paragraph under the title */
    .hero-description {
        position: relative;
        z-index: 2;
        margin-top: 6px;
        color: #bfc8d9;
        font-size: 18px;
        margin-left: 5%; /* Match title positioning */
        width: 45%;
        text-align: left;
    }

    /* Road container (used for cards, etc., not the main background) */
    .road-wrap {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .road {
        width: 80%;
        min-height: 320px;
        background:
            linear-gradient(90deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0) 15%),
            repeating-linear-gradient(180deg, #1b1d1f 0px, #1b1d1f 8px, #17181a 8px, #17181a 16px),
            linear-gradient(180deg, rgba(0,0,0,0.2), rgba(0,0,0,0.35));
        border-radius: 14px;
        box-shadow: 0 8px 40px rgba(2,6,10,0.7), inset 0 2px 0 rgba(255,255,255,0.015);
        position: relative;
        overflow: hidden;
    }

    /* yellow center divider */
    .road::before{
        content: "";
        position: absolute;
        left: 48%;
        width: 4%;
        top: 0;
        height: 100%;
        background: repeating-linear-gradient(
            to bottom,
            #f2c94c 0px,
            #f2c94c 14px,
            rgba(0,0,0,0) 14px,
            rgba(0,0,0,0) 28px
        );
        box-shadow: 0 0 20px rgba(242,201,76,0.06);
        transform: translateX(-50%);
    }

    /* dashed white lane markings on both sides */
    .road::after{
        content: "";
        position: absolute;
        left: 22px;
        right: 22px;
        top: 0;
        height: 100%;
        background:
          linear-gradient(180deg, rgba(0,0,0,0), rgba(0,0,0,0));
        background-image:
          repeating-linear-gradient(180deg, rgba(255,255,255,0.15) 0px, rgba(255,255,255,0.15) 6px, rgba(255,255,255,0) 6px, rgba(255,255,255,0) 24px);
        mask: linear-gradient(#000 0 0); /* ensure consistent dash opacity */
        opacity: 0.28;
        transform: translateX(0);
        pointer-events: none;
    }

    /* Card (glass) for uploader & results */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(1,6,10,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }

    /* Prediction result box (accent) */
    .result {
        background: linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.4));
        border-left: 6px solid rgba(242,201,76,0.95);
        padding: 14px;
        border-radius: 10px;
        color: #fff;
    }

    /* Confidence bar container */
    .conf-bar {
        height: 14px;
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        background: linear-gradient(90deg, #00c853, #f2c94c, #ff5252);
        transition: width 800ms ease;
    }

    /* Top5 list */
    .top5 {
        margin-top: 10px;
    }
    .top5 .row {
        display:flex;
        align-items:center;
        gap:10px;
        margin-bottom:8px;
    }
    .top5 .label {
        min-width: 32%;
        color: #e6eef8;
        font-size: 14px;
    }
    .top5 .bar {
        height:10px;
        border-radius:6px;
        background: rgba(255,255,255,0.06);
        width: 60%;
        overflow:hidden;
    }
    .top5 .bar > i {
        display:block;
        height:100%;
        background: linear-gradient(90deg,#00e676,#ffea00,#ff3d00);
    }

    /* Footer */
    .footer {
        text-align:center;
        color:#9fb0c9;
        padding-top:18px;
        padding-bottom:30px;
        font-size:14px;
    }

    /* Make uploader full width inside card */
    .uploader > div {
        width:100% !important;
    }

    /* Small screens */
    @media (max-width: 900px) {
        .hero-overlay-title { font-size: 36px; margin-left: 5%; width: 90%; text-align: center; }
        .hero-description { margin-left: 5%; width: 90%; text-align: center; }
        .hero-container { min-height: 250px; }
        .road { width: 94%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load model with caching
# -------------------------
@st.cache_resource
def load_tf_model():
    # Ensure 'model.h5' is in the same directory as this script
    return tf.keras.models.load_model("model.h5")

model = load_tf_model()

# -------------------------
# Class names (same as training)
# -------------------------
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

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image("https://raw.githubusercontent.com/streamlit/brand/master/streamlit-mark-light.png", width=120)
st.sidebar.title("Model Details")
st.sidebar.markdown(
    """
    **CNN Model** - Dataset: German Traffic Signs (GTSRB)  
    - Input: 32×32 RGB  
    - Classes: 43  
    - Norm: pixel/255.0  
    """
)
st.sidebar.caption("Tip: Use clear centered images (sign fills most of frame) for best results.")

# -------------------------
# Hero (custom background image area with overlaid title)
# -------------------------
st.markdown(
    """
    <div class="hero-container">
      <div class="hero-background-image"></div>
      <h1 class="hero-overlay-title">Traffic Sign Detection</h1>
      <p class="hero-description">Real-time sign classification with a road-style interface</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Main content area: left uploader / right results
# -------------------------
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload an image")
    uploader = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploader")
    st.markdown(
        "<div style='margin-top:8px; color:#c7d2e6'>Supported: JPG, PNG — try clear sign images</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Prediction")
    placeholder = st.empty()  # where dynamic content will appear
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Prediction flow
# -------------------------
if uploader is not None:
    # show preview in left column (re-use left_col to update UI)
    img = Image.open(uploader).convert("RGB")
    # display preview inside left card
    with left_col:
        st.image(img, caption="Uploaded image preview", use_column_width=True)

    # preprocess input
    x = img.resize((32, 32))
    arr = np.array(x, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # run model with spinner
    with st.spinner("🔎 Running model inference..."):
        preds = model.predict(arr)[0]
        idx_sorted = preds.argsort()[::-1]
        top5 = idx_sorted[:5]
        # main prediction
        main_idx = top5[0]
        main_label = class_names[int(main_idx)]
        main_conf = float(preds[main_idx])

    # fill right column placeholder with result card
    with right_col:
        # result HTML block
        conf_percent = main_conf * 100
        bar_width = f"{conf_percent:.2f}%"
        # top5 bars data
        top5_labels = [class_names[int(i)] for i in top5]
        top5_scores = [float(preds[int(i)]) for i in top5]
        # result markup
        result_html = f"""
        <div class="result" style="margin-bottom:12px;">
          <h3 style="margin:6px 0 4px 0;">✅ {main_label}</h3>
          <div style="color:#cbd8ea; font-size:14px; margin-bottom:8px;">
            Confidence: <strong style="color:#fff">{conf_percent:.2f}%</strong>
          </div>
          <div class="conf-bar" aria-hidden="true">
            <div class="conf-fill" style="width:{bar_width};"></div>
          </div>
        </div>
        """
        placeholder.markdown(result_html, unsafe_allow_html=True)

        # show Top-5 with bars
        rows_html = "<div class='top5'>"
        for label, score in zip(top5_labels, top5_scores):
            width_pct = int(score * 100)
            # ensure small visible width for tiny scores
            bar_html = f"<div class='row'><div class='label'>{label}</div>"
            bar_html += "<div class='bar'><i style='width:" + str(width_pct) + "%'></i></div>"
            bar_html += f"<div style='min-width:60px; text-align:right; color:#cbd8ea; margin-left:8px;'>{score*100:.1f}%</div></div>"
            rows_html += bar_html
        rows_html += "</div>"
        placeholder.markdown(rows_html, unsafe_allow_html=True)

        # numeric table (optional)
        df = pd.DataFrame({
            "label": [class_names[int(i)] for i in top5],
            "score": [float(preds[int(i)]) for i in top5]
        })
        st.markdown("<div style='margin-top:12px;'><small style='color:#9fb0c9'>Top-5 numeric (for debugging)</small></div>", unsafe_allow_html=True)
        st.table(df.style.format({"score": "{:.4f}"}))

else:
    # show guidance when no image
    with left_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Demo Tips")
        st.markdown("- Use clear photos with the sign centered\n- Avoid heavy motion blur\n- Prefer square-ish crops\n")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.info("Upload an image to get predictions. Try the sample traffic signs for best results.")

# -------------------------
# Footer
# -------------------------
st.markdown("<div class='footer'>🚦 Built with TensorFlow & Streamlit — Developed by Atharv</div>", unsafe_allow_html=True)
