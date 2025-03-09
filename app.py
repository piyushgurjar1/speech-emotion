import os
import numpy as np
import librosa
from flask import Flask, render_template, request
from tensorflow.python.keras.models import load_model

import tensorflow as tf

app = Flask(__name__)

# Custom model loader with proper input shape handling
def custom_load_model(filepath):
    # Load model architecture with custom objects
    with open(filepath, 'r') as f:
        model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                # 'l1_l2': regularizers.l1_l2,
                'Adam': tf.keras.optimizers.Adam
            }
        )
    # return load_model(filepath)
    return model

# Load model (use absolute path for better reliability)
MODEL_PATH = os.path.join('models', 'speech_emotion_model.h5')
model = custom_load_model(MODEL_PATH)

# IMPORTANT: Emotion labels must match LabelEncoder's alphabetical sorting from training
emotion_mapping = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_feature(data, sr):
    """Identical feature extraction to training code"""
    result = np.array([])
    
    # MFCC (40 features)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    
    # Chroma (12 features)
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))
    
    # Mel Spectrogram (128 features)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('audio')
        if file and file.filename.endswith('.wav'):
            try:
                # Save temporary file
                temp_dir = 'temp'
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                
                # Process audio identically to training pipeline
                data, sr = librosa.load(file_path, sr=None)
                features = extract_feature(data, sr)
                
                # Ensure exact feature length (180)
                if len(features) != 180:
                    features = np.pad(features, (0, max(0, 180 - len(features))))[:180]
                
                # Reshape for model input (batch_size=1, steps=180, channels=1)
                input_data = features.reshape(1, 180, 1)
                
                # Make prediction
                preds = model.predict(input_data)
                prediction = emotion_mapping[np.argmax(preds)]
                
            except Exception as e:
                prediction = f"Processing error: {str(e)}"
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            prediction = "Invalid file format - please upload a WAV file"
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
 