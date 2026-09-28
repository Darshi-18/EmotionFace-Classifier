import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Set up page configurations
st.set_page_config(page_title="EmotionFace Classifier", layout="centered")
st.title("🧠 EmotionFace: Facial Expression Classifier")
st.write("Choose your preferred input method below to classify emotional expressions.")

# 1. Load the trained model weights safely
MODEL_PATH = 'best_emotion_model.h5'

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Could not find '{MODEL_PATH}' file in this directory!")
        return None
    return load_model(MODEL_PATH, compile=False)

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 2. Input Mode Selection: Let users choose between Uploading or Webcam
input_mode = st.radio("Select Input Method:", ("Upload an Image File", "Use Live Webcam Cam"))

img_data = None

if input_mode == "Upload an Image File":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open uploaded image using PIL and display a preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        # Convert PIL image to an OpenCV compatible numpy array format
        img_data = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

else:
    img_file_buffer = st.camera_input("Capture your face to run the system:")
    if img_file_buffer is not None:
        # Read the webcam snapshot buffer array
        bytes_data = img_file_buffer.getvalue()
        img_data = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# 3. Model Classification Execution Engine
if img_data is not None and model is not None:
    # Preprocess image structure to match model input parameters (Grayscale, 48x48)
    gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (48, 48))
    
    # Scale array dimensions to match batch size requirements: (1, 48, 48, 1)
    img_pixels = np.expand_dims(resized_img, axis=0)
    img_pixels = np.expand_dims(img_pixels, axis=-1)
    img_pixels = img_pixels / 255.0  # Normalize intensity
    
    # Run the model classification
    predictions = model.predict(img_pixels, verbose=0)
    
    # Extract index and score directly from the first element of the batch output row
    max_index = int(np.argmax(predictions[0]))
    predicted_emotion = emotion_labels[max_index]
    confidence_score = float(predictions[0][max_index]) * 100
    
    # Display results to the web screen user panel
    st.success(f"### Predicted Emotion: **{predicted_emotion}**")
    st.metric(label="Prediction Confidence", value=f"{confidence_score:.2f}%")
    # Create a clean probability dictionary for all emotions
    prob_dict = {emotion_labels[i]: float(predictions[0][i]) for i in range(len(emotion_labels))}
    
    # Display a beautiful horizontal bar chart of the distributions
    st.write("### 📊 Emotion Probability Distribution")
    st.bar_chart(prob_dict)

