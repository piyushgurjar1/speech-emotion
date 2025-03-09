import os
import numpy as np
import librosa
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Simplified model loading
model = tf.keras.models.load_model(os.path.join('models', 'speech_emotion_model.h5'))

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def process_audio(data, sr):
    """Optimized feature extraction with fixed size output"""
    features = []
    
    # MFCC (40 coefficients)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfccs, axis=1))
    
    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=data, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    features.extend(np.mean(mel, axis=1))
    
    # Ensure exactly 180 features
    return np.pad(features, (0, 180))[:180]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST' and 'audio' in request.files:
        file = request.files['audio']
        if file.filename.endswith('.wav'):
            try:
                # Process audio without saving to disk
                audio, sr = librosa.load(file.stream, sr=None)
                features = process_audio(audio, sr)
                prediction = model.predict(features.reshape(1, 180, 1))
                result = emotion_labels[np.argmax(prediction)]
            except Exception as e:
                result = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    pass
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))