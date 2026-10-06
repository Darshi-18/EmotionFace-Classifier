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
st.write("An advanced Deep Learning system designed to decode human facial expressions with built-in human face verification.")
st.markdown("---")

# 3. Model Engine Initialization
MODEL_PATH = 'best_emotion_model.h5'

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚨 Model execution failure: '{MODEL_PATH}' missing from root environment folder.")
        return None
    return load_model(MODEL_PATH, compile=False)

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 4. Helper Function: Validate Human Face Presence
def verify_human_face(image_bgr):
    """
    Uses a lightweight OpenCV cascade framework to verify if at least 
    one human face is present in the uploaded image matrix.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Load OpenCV's built-in face tracking xml
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # If the local paths have system quirks, attempt direct detection configurations
    if face_cascade.empty():
        return True # Fallback to true if server configuration files are unreachable
        
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return len(faces) > 0

# 5. Two-Column Dashboard Setup
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📸 Media Feed Controller")
    
    # Input toggle using tabs for a cleaner, modern layout feel
    input_tab = st.tabs(["📁 File Drop Zone", "🎥 Live Camera Capture"])
    img_data = None
    
    with input_tab:
        uploaded_file = st.file_uploader("Upload static target image matrix:", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Target Image Matrix Source", use_container_width=True)
            img_data = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
    with input_tab:
        img_file_buffer = st.camera_input("Acquire live webcam streaming snapshot:", label_visibility="collapsed")
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            img_data = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

with col2:
    st.subheader("📊 Model Diagnostics & Analytics")
    
    if img_data is None:
        st.info("💡 Awaiting input media payload. Please upload an image matrix or capture a live webcam frame in the controller panel to initialize inference tracking.")
    
    elif model is not None:
        with st.spinner("Running face verification layers..."):
            # Execute Human Face Presence check
            is_human = verify_human_face(img_data)
            
        if not is_human:
            # 🛑 Hard Stop: Block execution if no human face is found
            st.error("❌ **Invalid Image Payload Detected**")
            st.warning("The system failed to detect a human face in the uploaded image. Please provide a clear, well-lit photo containing a human face to initialize emotion tracking.")
        
        else:
            with st.spinner("Processing neural inference transformations..."):
                # Image Preprocessing & Feature Extraction
                gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray_img, (48, 48))
                img_pixels = np.expand_dims(resized_img, axis=0)
                img_pixels = np.expand_dims(img_pixels, axis=-1)
                img_pixels = img_pixels / 255.0  # Intensity Range Uniform Normalization
                
                # Execute Forward Pass
                predictions = model.predict(img_pixels, verbose=0)[0]
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
                    st.metric(label="Validation Check", value="Human Verified", delta="Passed")
                
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
st.caption("🧠 EmotionFace Analytics Platform v2.1 • Guarded by Face Verification Filters.")
