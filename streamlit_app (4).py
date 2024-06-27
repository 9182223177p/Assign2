import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('lstm_sign_language_model.h5')

IMG_HEIGHT = 64
IMG_WIDTH = 64
MAX_FRAMES = 40

def get_video_frames(video_path, max_frames=40):
    capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while capture.isOpened() and frame_count < max_frames:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        frames.append(frame)
        frame_count += 1
    capture.release()
    return np.array(frames)


def preprocess_image(image):
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

st.set_page_config(page_title="Sign Language Recognizer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #4a90e2;
        }
        .stButton>button {
            color: white;
            background-color: #4a90e2;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)


st.title("Sign Language Recognition System")
st.write("Upload a video or image to recognize sign language.")

uploaded_file = st.file_uploader("Choose a video or image file", type=["mp4", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

   
    if uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
        
        temp_video_path = 'temp_video.mp4'
        with open(temp_video_path, 'wb') as temp_file:
            temp_file.write(file_bytes)

        st.write("Processing video...")

       
        frames = get_video_frames(temp_video_path, MAX_FRAMES)
        if frames.shape[0] == MAX_FRAMES:
            st.write("Video processed successfully.")
        else:
            st.write("Video does not contain enough frames.")

   
    elif uploaded_file.type in ["image/png", "image/jpg", "image/jpeg"]:
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image")
        
        processed_image = preprocess_image(image)
        st.write("Image processed successfully.")

    if model is not None:
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        st.write(f"Predicted Class: {predicted_class}")

st.sidebar.title("About")
st.sidebar.info("This application recognizes sign language from videos and images using a trained LSTM model.")

st.sidebar.title("Instructions")
st.sidebar.info("""
- Upload a video (.mp4) or an image (.png, .jpg, .jpeg).
- The system will process the file and attempt to recognize the sign language.
- Ensure your video has enough frames for better accuracy.
""")
