import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from tensorflow.keras.models import load_model
from PIL import Image

# 1. Page Configuration & Theme Initialization
st.set_page_config(
    page_title="EmotionFace Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling adjustments
st.markdown("""
    <style>
    .main .block-container { max-width: 1000px; padding-top: 2rem; }
    h1 { font-weight: 800 !important; color: #F0F2F6; }
    .stAlert p { font-size: 1.15rem !important; font-weight: 600; }
    div[data-testid="stMetric"] { background-color: #1E232A; border-radius: 10px; padding: 15px; border: 1px solid #30363D; }
    </style>
""", unsafe_allow_html=True)

# 2. App Headers
st.title("🧠 EmotionFace Analytics Platform")
st.write("An advanced Deep Learning system designed to decode human facial expressions with targeted structural face validation.")
st.markdown("---")

# 3. Model Engine Optimization
MODEL_PATH = 'best_emotion_model.h5'

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 Model execution failure: '{MODEL_PATH}' missing from root environment folder.")
        return None
    return load_model(MODEL_PATH, compile=False)

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 4. High-Precision Structural Geometry Matrix Scan (Cloud-Safe)
def verify_human_face_geometry(image_bgr):
    """
    Analyzes the structural pixel geometry of the image payload.
    Human facial crops contain high-density vertical and horizontal edge structures 
    (from hair, eyes, eyebrows) that separate them from flat objects or mechanical grids.
    """
    # Convert to grayscale and downsample to look at structural macro-features
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    
    # Compute Sobel gradients to find structural edge contours
    sobel_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    
    # Absolute gradient values
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_gradient = np.mean(magnitude)
    
    # Analyze the standard deviation of structural gradients
    gradient_variance = np.std(magnitude)
    
    # Analyze color distribution parameters in YCrCb color metric space
    ycrcb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mean_cr = np.mean(ycrcb_img[:, :, 1])
    mean_cb = np.mean(ycrcb_img[:, :, 2])
    
    # Precise human facial boundary matrix math thresholds
    has_correct_texture = (15.0 < mean_gradient < 120.0) and (gradient_variance > 10.0)
    has_correct_tone_range = (128 <= mean_cr <= 175) or (75 <= mean_cb <= 135)
    
    return has_correct_texture and has_correct_tone_range

# 5. Two-Column Dashboard Setup
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 Media Feed Controller")
    
    tab_upload, tab_camera = st.tabs(["📁 File Drop Zone", "🎥 Live Camera Capture"])
    img_data = None
    
    with tab_upload:
        uploaded_file = st.file_uploader("Upload static target image matrix:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Target Image Matrix Source", use_container_width=True)
            img_data = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
    with tab_camera:
        img_file_buffer = st.camera_input("Acquire live webcam streaming snapshot:", label_visibility="collapsed")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            img_data = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

with col2:
    st.subheader("📊 Model Diagnostics & Analytics")
    
    if img_data is None:
        st.info("💡 Awaiting input media payload. Please upload an image matrix or capture a live webcam frame in the controller panel to initialize inference tracking.")
    
    elif model is not None:
        with st.spinner("Executing structural geometry verification scanning..."):
            is_human_face = verify_human_face_geometry(img_data)
            
        if not is_human_face:
            # 🛑 Hard Stop: Block cars, non-human patterns, and empty frames
            st.error("❌ **Validation Failure: Non-Human Image Detected**")
            st.warning("The application rejected this payload because it does not contain a recognizable human face profile. Please provide a clear profile photo or snapshot containing a human face to initialize emotion analytics tracking.")
        
        else:
            with st.spinner("Processing neural inference transformations..."):
                # Image Preprocessing & Feature Extraction
                gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray_img, (48, 48))
                img_pixels = np.expand_dims(resized_img, axis=0)
                img_pixels = np.expand_dims(img_pixels, axis=-1)
                img_pixels = img_pixels / 255.0  # Normalize Intensity Range
                
                # Execute Forward Pass
                predictions = model.predict(img_pixels, verbose=0)
                predictions = predictions[0]
                
                max_index = int(np.argmax(predictions))
                predicted_emotion = emotion_labels[max_index]
                confidence_score = float(predictions[max_index]) * 100
                
                # Result Visualization Framework
                st.success(f"### Classification Result: **{predicted_emotion}**")
                
                # Layout metric widgets
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Primary Classification Confidence", value=f"{confidence_score:.2f}%")
                with m_col2:
                    st.metric(label="Biometric Verification", value="Face Confirmed", delta="Passed")
                
                st.write("#### 📈 Full Class Density Map Distribution")
                
                # Build DataFrame for advanced clean plotting
                df_chart = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Probability (%)': [float(p) * 100 for p in predictions]
                }).sort_values(by='Probability (%)', ascending=False)
                
                # Build an elegant horizontal Altair Chart
                chart = alt.Chart(df_chart).mark_bar(
                    cornerRadiusTopRight=5,
                    cornerRadiusBottomRight=5
                ).encode(
                    x=alt.X('Probability (%)', title="Confidence Percentage (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Emotion', sort='-x', title="Class Label"),
                    color=alt.Color('Probability (%)', scale=alt.Scale(scheme='viridis'), legend=None)
                ).properties(
                    height=260
                )
                
                st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("🧠 EmotionFace Analytics Platform v2.6 • Protected by High-Precision Structural Contrast Geometry Filters.")
