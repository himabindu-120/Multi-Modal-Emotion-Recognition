from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from fer import FER
import cv2
import librosa
import numpy as np
import math
import moviepy.editor as mp
import tensorflow as tf
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Constants and Blueprint Setup
views = Blueprint('views', __name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
emotion_detector = FER()
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load Audio Emotion Model (replace with your actual path)
model3_path = r"D:\\emotion\\website\\model3.h5"
emotion_labels = {
    0: 'fear', 1: 'disgust', 2: 'neutral', 3: 'happy', 4: 'sadness', 5: 'surprise', 6: 'angry'
}

def load_model_custom(path, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}
    custom_objects['SpatialDropout1D'] = tf.keras.layers.SpatialDropout1D
    return tf.keras.models.load_model(path, custom_objects=custom_objects)

try:
    model3 = load_model_custom(model3_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model3_path}")
    model3 = None

# Load the text emotion model components
text_model_path = r"D:\\emotion\\website\\emotion_classification_model.h5"
encoder_path = r"D:\\emotion\\website\\encoder.pkl"
cv_path = r"D:\\emotion\\website\\CountVectorizer.pkl"

# Load the encoder
try:
    with open(encoder_path, "rb") as encoder_file:
        loaded_encoder = pickle.load(encoder_file)
except Exception as e:
    print(f"Error loading the encoder: {e}")
    loaded_encoder = None

# Load the CountVectorizer
try:
    with open(cv_path, "rb") as cv_file:
        loaded_cv = pickle.load(cv_file)
except Exception as e:
    print(f"Error loading the CountVectorizer: {e}")
    loaded_cv = None

# Preprocess function for text
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    return " ".join(review)

def predict_text_emotion(text_data):
    preprocessed_text = preprocess_text(text_data)
    X_test = loaded_cv.transform([preprocessed_text]).toarray()
    predictions = text_model_path.predict(X_test)
    labels = loaded_encoder.classes_
    predicted_label_indices = np.argmax(predictions, axis=1)
    predicted_labels = loaded_encoder.inverse_transform(predicted_label_indices)
    accuracies = np.max(predictions, axis=1)
    return list(zip(predicted_labels, accuracies))

def process_audio_chunk(chunk, length_chosen=130307):
    if len(chunk) > length_chosen:
        new_chunk = chunk[:length_chosen]
    elif len(chunk) < length_chosen:
        pad_width = math.ceil((length_chosen - len(chunk)) / 2)
        new_chunk = np.pad(chunk, (pad_width, pad_width), mode='median')
    else:
        new_chunk = chunk
    mfcc = librosa.feature.mfcc(y=new_chunk, sr=44000, n_mfcc=40)
    mfcc = mfcc.T
    mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
    probabilities = model3.predict(mfcc)
    dominant_emotion_index = np.argmax(probabilities)
    dominant_emotion_label = emotion_labels[dominant_emotion_index]
    accuracy = probabilities[0, dominant_emotion_index]
    return (dominant_emotion_label, accuracy)

def process_video(video_path):
    capture = cv2.VideoCapture(video_path)
    detected_emotions = []
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    sample_rate = audio_clip.fps
    duration = audio_clip.duration
    total_frames = int(sample_rate * duration)
    predicted_audio_emotions = []

    for i in range(0, total_frames, 44100):
        ret, frame = capture.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_model.detectMultiScale(gray_frame, 1.1, 5)
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            emotions = emotion_detector.detect_emotions(face_roi)
            if emotions:
                dominant_emotion = emotions[0]["emotions"]
                dominant_emotion_name = max(dominant_emotion, key=dominant_emotion.get)
                accuracy = dominant_emotion[dominant_emotion_name]
                detected_emotions.append((dominant_emotion_name, accuracy))

        audio_chunk = audio_clip.subclip(i / sample_rate, (i + 44100) / sample_rate)
        audio_chunk = audio_chunk.to_soundarray()
        audio_chunk = audio_chunk.mean(axis=1)
        predicted_audio_emotion, audio_accuracy = process_audio_chunk(audio_chunk)
        predicted_audio_emotions.append((predicted_audio_emotion, audio_accuracy))

    capture.release()
    cv2.destroyAllWindows()
    return detected_emotions, predicted_audio_emotions

@views.route('/', methods=['GET', 'POST'])
@views.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/result')
@login_required
def result():
    results = {}  
    return render_template("result.html", results=results)

@views.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        detected_emotions, predicted_audio_emotions = process_video(filepath)

        if detected_emotions or predicted_audio_emotions:
            results = {
                'video_emotion': detected_emotions,
                'audio_emotion': predicted_audio_emotions,
                'video_filename': os.path.join('uploads', filename),
            }
            return render_template('result.html', results=results, emotion_labels=emotion_labels)

    return redirect(request.url)




