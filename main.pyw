import numpy as np
import librosa
import librosa.display
import base64
import os
import io
import tkinter as tk
import pygame
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from tkinter import filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle

pygame.mixer.init()

# Icon Base64
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


# Initialize or load the model and label encoder
def load_or_initialize_model():
    global model, le, X_train, y_train

    # Initialize model and label encoder
    model = SVC(kernel='linear')
    le = LabelEncoder()

    # Check if model and label encoder files exist
    if os.path.exists("tone_model.pkl") and os.path.exists("label_encoder.pkl"):
        with open("tone_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        # Load training data if available
        if os.path.exists("X_train.npy") and os.path.exists("y_train.npy"):
            X_train = np.load("X_train.npy")
            y_train = np.load("y_train.npy")
        else:
            X_train = np.array([])
            y_train = np.array([])
    else:
        X_train = np.array([])
        y_train = np.array([])

    # If the model is not trained yet, use initial data to train
    if X_train.size == 0:
        print("No initial training data found. Training with predefined data.")
        # Predefined initial training data
        initial_audio_files = ['neutral.wav', 'happy.wav', 'sad.wav', 'mad.wav']
        initial_labels = ['neutral', 'happy', 'sad', 'mad']

        data = []
        labels = []
        for audio, label in zip(initial_audio_files, initial_labels):
            if os.path.exists(audio):
                features = extract_features(audio)
                data.append(features)
                labels.append(label)
            else:
                print(f"File {audio} not found!")

        if data:
            X_train = np.array(data)
            y_train = np.array(labels)

            # Encode labels
            le.fit(y_train)
            y_encoded = le.transform(y_train)

            # Train the model
            model.fit(X_train, y_encoded)

            # Save the initial model and label encoder
            with open("tone_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)

            np.save("X_train.npy", X_train)
            np.save("y_train.npy", y_train)


# Function to predict the tone of a new audio file
def predict_tone(audio_path):
    global X_train, y_train

    if os.path.exists(audio_path):
        if X_train.size == 0:
            return "Model not trained yet. Please provide some initial data."
        features = extract_features(audio_path)
        features = features.reshape(1, -1)  # Reshape for a single prediction
        prediction = model.predict(features)
        return le.inverse_transform(prediction)[0]
    else:
        return "File not found!"


# Function to update the model with correct tone
def update_model(audio_path, correct_tone):
    global X_train, y_train

    features = extract_features(audio_path)
    X_train = np.vstack([X_train, features]) if X_train.size else features.reshape(1, -1)
    y_train = np.append(y_train, correct_tone)

    le.fit(y_train)  # Refit label encoder with updated labels
    y_encoded = le.transform(y_train)

    model.fit(X_train, y_encoded)

    with open("tone_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)


# Function to plot the waveform
def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    return fig


# Function to play the audio
def play_audio(audio_path):
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()


# Function to open file dialog and display the tone and waveform
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        tone = predict_tone(file_path)
        result_label.config(text=f"Predicted Tone: {tone}")

        # Plot and display waveform
        fig = plot_waveform(file_path)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)
        play_audio(file_path)

        # Ask user if the prediction was correct
        response = messagebox.askquestion("Confirm Prediction", f"Is the prediction '{tone}' correct?")

        if response == 'no':
            correct_tone = simpledialog.askstring("Correct Tone", "Please enter the correct tone:")
            if correct_tone:
                update_model(file_path, correct_tone)
                messagebox.showinfo("Model Updated", "The model has been updated with the correct tone.")
            else:
                messagebox.showwarning("Input Error", "No correct tone provided. The model was not updated.")
        else:
            messagebox.showinfo("Confirmation",
                                "The prediction was confirmed as correct. No updates were made to the model.")


# Set up the GUI
window = tk.Tk()
window.title("Voice Tone Detector")
window.geometry("600x500")

# Set up the icon
icon = get_icon_from_base64(ICON_BASE64)
window.iconphoto(False, icon)

# Add widgets
open_button = tk.Button(window, text="Open Audio File", command=open_file)
open_button.pack(pady=20)

result_label = tk.Label(window, text="Predicted Tone: None")
result_label.pack(pady=20)

# Load or initialize the model
load_or_initialize_model()

window.mainloop()
