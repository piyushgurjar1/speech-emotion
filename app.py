import os
import numpy as np
import librosa
from flask import Flask, render_template, request
import tensorflow as tf  # Changed import

app = Flask(__name__)

# Optimized model loading
def load_emotion_model():
    model_path = os.path.join('models', 'speech_emotion_model.h5')
    return tf.keras.models.load_model(model_path)

model = load_emotion_model()

# Verified emotion mapping order
emotion_mapping = ['angry', 'calm', 'disgust', 'fearful', 
                   'happy', 'neutral', 'sad', 'surprised']

# Optimized feature extraction
def extract_feature(data, sr):
    features = []
    
    # MFCC (librosa 0.10.1 compatible)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    features.extend(mfccs)
    
    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features.extend(chroma)
    
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    features.extend(mel)
    
    return np.array(features)[:180]  # Ensure 180 features

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('audio')
        if file and file.filename.endswith('.wav'):
            try:
                # In-memory processing (no temp files)
                data, sr = librosa.load(file.stream, sr=None)
                features = extract_feature(data, sr)
                
                # Pad/truncate to exactly 180 features
                features = np.pad(features, (0, 180))[:180]
                
                # Reshape for model input
                input_data = features.reshape(1, 180, 1)
                
                # Prediction
                preds = model.predict(input_data)
                prediction = emotion_mapping[np.argmax(preds)]
                
            except Exception as e:
                prediction = f"Error: {str(e)}"
                
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    True
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))