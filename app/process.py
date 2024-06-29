import os
import cv2
import numpy as np
from tensorflow import keras




# Function to load and preprocess a single test video
def load_test_video(video_path, frames_per_video=10, img_height=64, img_width=64):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (img_width, img_height))
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Function to predict the class of a test video
def predict_video_class(video_path):
    loaded_model = keras.models.load_model('backend/video_classification_model2.h5')

    class_names = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Labels', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']  # Modify this based on your actual class names

    img_height, img_width = 64, 64
    frames_per_video = 10
    test_video = load_test_video(video_path, frames_per_video)
    test_video = test_video.reshape(-1, frames_per_video, img_height, img_width, 3)
    predictions = loaded_model.predict(test_video)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    return predicted_class
