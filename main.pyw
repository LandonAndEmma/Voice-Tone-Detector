import numpy as np
import librosa
import base64
import os
import io
import tkinter as tk
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tkinter import filedialog, messagebox


ICON_BASE64 = """iVBORw0KGgoAAAANSUhEUgAAAIAAAACABAMAAAAxEHz4AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAwUExURQAAAFJKQtYhMWtjY+8pOoR7e/daY5ycnPeMjK2trb29vf+9pc7Ozt7e3u/v7/f396N/qJUAAAABdFJOUwBA5thmAAADMUlEQVR4nO2WT2jTUBzHc3iHOZqyYEQYuELZSTw4POlB0vFadLOUHorMyzyMIchgBy+C8yQelIFKPWQg9LAxGlq8DBF6UeyUjQRWNhmWpN1hIsgGO/Q4re9P+ie1XV6abVXIN4cmTb+fl9/3995LOc6TJ0+ePP0f4o3yQee7oDsAaHPWSf7vceHs9Z1jBnCSVD+zA/AFASlw0B4gcdApgDoh4PCBPm1roIB6EZKEBwdE6ALah9AMQFYIsZOYkSCQ7ELw7xOAMJgiAEiHNu0QcLYpNgPI0KBuBrR+mxp4/dMIqeEyMVvswL6HVoDFTK7s/RxfTL8kgGsprtXO4m8GWM2YxeI3ilmaohCw2Gk7HQBEIVBpGhsAVj8SBYjnJqWWEB0BhFB4srl/jBk2AKFQKNAaojMAhGLFcYbEr6/GcQRIi/WxAWOIVsBgzNoBiclf/v1jbvPK1eRrFEKsPjZpgSOAHA6hFCqWWWxbAogif7V6qH1NTMg34QNR3GqEaDt6KyCZRICykw4ACKPYXz1U19MyAmwJOISmncQZQI7iFB3MYuSHtzQE+GVsP1LXZDmGQhzaZd9J/gIkK6I4tEvMLADsH383U9o36FyaWFhEgMEU6aDEFmEL4C3KYCwlMbyKqH88l8utRInd2H745tvCMgLAAcYOtAVAtBwG8PSTGOyR+9gPTYCuapsjeEHhd6y9vSMAbQqtb/lOgMi95++fTk/BG/MmYX0pTva1oZ3TANACVlAfYWTGBHyZozvrhY2TB/j38uOmHeLlhB5fQcrqHzHg/GPbGtwBeF1V1XxkCtY1plAts9XgFuBTsT7X/KPBYPCiYiWcCkDNm2as4ahOvltXMlrhNAHEP1zrpKFr+Ci0+c98vIC+ZxkCWHtFzZHpu/QGb67pwgkD+p6MXiJxKcoiDCcSiduzlvv+Pc0wNo56ALcADkg+5M6oq4nEHTx3Ukf9+EQAWD71Q/8SsaflFz0BYJ1ZoOsn3S3BNaC/toJnPUDvAV0TXAP8PxV3M8k14B9oA19U3BXhGtBoZO8BPZ+LXc8k1wBaQ7ZcPegZwKcZpf3STrf2YwB48uTJk6de6g9m4hJWGljwVAAAAABJRU5ErkJggg=="""


def get_icon_from_base64(base64_string):
    icon_data = base64.b64decode(base64_string)
    icon = Image.open(io.BytesIO(icon_data))
    return ImageTk.PhotoImage(icon)


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
root.geometry("600x400")
root.iconphoto(True, get_icon_from_base64(ICON_BASE64))


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
