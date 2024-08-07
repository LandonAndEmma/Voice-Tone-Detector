import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import tkinter as tk
from tkinter import filedialog, messagebox


# Function to extract features from an audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, mel, contrast])


# Example dataset
audio_files = ['neutral.wav', 'happy.wav', 'sad.wav', 'mad.wav']
audio_labels = ['neutral', 'happy', 'sad', 'mad']

# Extract features and labels
data = []
labels = []

for audio, label in zip(audio_files, audio_labels):
    if os.path.exists(audio):
        features = extract_features(audio)
        data.append(features)
        labels.append(label)
    else:
        print(f"File {audio} not found!")

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Support Vector Classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Tkinter GUI setup
root = tk.Tk()
root.title("Voice Tone Detector")


# Function to predict the tone of a new audio file
def predict_tone(audio_path):
    if os.path.exists(audio_path):
        features = extract_features(audio_path)
        features = features.reshape(1, -1)  # Reshape for a single prediction
        prediction = model.predict(features)
        return le.inverse_transform(prediction)[0]
    else:
        return "File not found!"


# Function to open file dialog and display the tone
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        tone = predict_tone(file_path)
        result_label.config(text=f"Predicted Tone: {tone}")
    else:
        messagebox.showwarning("Warning", "No file selected")


# Tkinter widgets
open_button = tk.Button(root, text="Open Audio File", command=open_file)
open_button.pack(pady=20)

result_label = tk.Label(root, text="Predicted Tone: None")
result_label.pack(pady=20)

# Start the Tkinter main loop
root.mainloop()
