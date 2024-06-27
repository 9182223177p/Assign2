import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

IMG_HEIGHT = 64
IMG_WIDTH = 64
MAX_FRAMES = 40
DATA_DIR = './WLSAL'
ANNOTATION_FILE = 'WLASL_v0.3.json'

with open(os.path.join(DATA_DIR, ANNOTATION_FILE), 'r') as file:
    annotations = json.load(file)

video_files = []
class_labels = []

for annotation in annotations:
    gloss_word = annotation['gloss']
    instances = annotation['instances']
    for instance in instances:
        vid_id = instance['video_id']
        vid_path = os.path.join("./wlasl-processed", 'videos', f"{vid_id}.mp4")
        if os.path.exists(vid_path):
            video_files.append(vid_path)
            class_labels.append(gloss_word)

def fetch_frames(video_file, max_frames=40):
    capture = cv2.VideoCapture(video_file)
    frame_list = []
    frame_count = 0
    while capture.isOpened() and frame_count < max_frames:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
        frame_list.append(frame)
        frame_count += 1
    capture.release()
    return np.array(frame_list)


X_data = []
y_data = []

for video_file, label in zip(video_files, class_labels):
    frames = fetch_frames(video_file, MAX_FRAMES)
    if frames.shape[0] == MAX_FRAMES:
        X_data.append(frames)
        y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)
print(len(y_data))

try:
    label_encoder = LabelEncoder()
    y_data_encoded = label_encoder.fit_transform(y_data)

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data_encoded, test_size=0.2, random_state=42)
except:
    pass


num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

lstm_model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(MAX_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 10
history = lstm_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))


model_save_path = 'lstm_sign_language_model.h5'
lstm_model.save(model_save_path)


