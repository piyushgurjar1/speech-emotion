import os
import numpy as np
import librosa
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Optimized model loading
model = tf.keras.models.load_model(os.path.join('models', 'speech_emotion_model.h5'))

emotion_mapping = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_feature(data, sr):
    """Optimized feature extraction"""
    # Explicitly use float32 to match training
    data = data.astype(np.float32)
    
    # Compute features directly without intermediate arrays
    return np.concatenate([
        np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0),
        np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    ])[:180]  # Ensure exactly 180 features

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('audio')
        if file and file.filename.endswith('.wav'):
            try:
                # Process audio without saving to disk
                data, sr = librosa.load(file.stream, sr=None)
                features = extract_feature(data, sr)
                
                # Zero-pad features to ensure 180 dimensions
                features = np.pad(features, (0, 180))[:180]
                
                # Reshape for model input
                input_data = features.reshape(1, 180, 1).astype(np.float32)
                
                # Predict
                preds = model.predict(input_data)
                prediction = emotion_mapping[np.argmax(preds)]
                
            except Exception as e:
                prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    True
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))