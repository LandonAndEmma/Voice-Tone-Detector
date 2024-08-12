import base64
import io
import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

pygame.mixer.init()
ICON_BASE64 = """iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAS1BMVEUAAADvjJS9vb1SSkLm5ube3t7Ozs7vKTrFxcXvvcWcnJzelJTvWmtaUlLeITHv7+/W1ta1tbWtra2lpaX3lJT3WmOtSkr3QkqcOkLJAJcIAAAAAXRSTlMAQObYZgAAAFFJREFUGNOljDcOgEAQA232EpfI4f8vRULao6CD6WZkGd+RAQTZq1spmcxUN6bsKwnFJzNO57M/Nm+tiKpzvlYX5hZSiPH+QaProGhYXgV/uAAQeQHIXWPCWwAAAABJRU5ErkJggg=="""


def get_icon_from_base64(base64_string):
    icon_data = base64.b64decode(base64_string)
    icon = Image.open(io.BytesIO(icon_data))
    return ImageTk.PhotoImage(icon)


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast])


def load_or_initialize_model():
    global model, le, X_train, y_train
    model = SVC(kernel="linear")
    le = LabelEncoder()
    if os.path.exists("tone_model.pkl") and os.path.exists("label_encoder.pkl"):
        with open("tone_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        if os.path.exists("X_train.npy") and os.path.exists("y_train.npy"):
            X_train = np.load("X_train.npy")
            y_train = np.load("y_train.npy")
        else:
            X_train = np.array([])
            y_train = np.array([])
    else:
        X_train = np.array([])
        y_train = np.array([])
    if X_train.size == 0:
        print("No initial training data found. Training with predefined data.")
        initial_audio_files = ["neutral.wav", "happy.wav", "sad.wav", "mad.wav"]
        initial_labels = ["neutral", "happy", "sad", "mad"]
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
            le.fit(y_train)
            y_encoded = le.transform(y_train)
            model.fit(X_train, y_encoded)
            with open("tone_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
            np.save("X_train.npy", X_train)
            np.save("y_train.npy", y_train)


def predict_tone(audio_path):
    global X_train, y_train
    if os.path.exists(audio_path):
        if X_train.size == 0:
            return "Model not trained yet. Please provide some initial data."
        features = extract_features(audio_path)
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return le.inverse_transform(prediction)[0]
    else:
        return "File not found!"


def update_model(audio_path, correct_tone):
    global X_train, y_train
    features = extract_features(audio_path)
    X_train = (
        np.vstack([X_train, features]) if X_train.size else features.reshape(1, -1)
    )
    y_train = np.append(y_train, correct_tone)
    le.fit(y_train)
    y_encoded = le.transform(y_train)
    model.fit(X_train, y_encoded)
    with open("tone_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)


def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    return fig


def play_audio(audio_path):
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()


def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        tone = predict_tone(file_path)
        result_label.config(text=f"Predicted Tone: {tone}")
        fig = plot_waveform(file_path)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)
        play_audio(file_path)
        response = messagebox.askquestion(
            "Confirm Prediction", f"Is the prediction '{tone}' correct?"
        )
        if response == "no":
            correct_tone = simpledialog.askstring(
                "Correct Tone", "Please enter the correct tone:"
            )
            if correct_tone:
                update_model(file_path, correct_tone)
                messagebox.showinfo(
                    "Model Updated", "The model has been updated with the correct tone."
                )
            else:
                messagebox.showwarning(
                    "Input Error",
                    "No correct tone provided. The model was not updated.",
                )
        else:
            messagebox.showinfo(
                "Confirmation",
                "The prediction was confirmed as correct. No updates were made to the model.",
            )


window = tk.Tk()
window.title("Voice Tone Detector")
window.geometry("600x500")
window.minsize(275, 100)
icon = get_icon_from_base64(ICON_BASE64)
window.iconphoto(False, icon)
menubar = tk.Menu(window)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
menubar.add_cascade(label="File", menu=file_menu)
help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="Help")
help_menu.add_command(label="Repository")
menubar.add_cascade(label="Help", menu=help_menu)
window.config(menu=menubar)
result_label = tk.Label(window, text="Predicted Tone: None")
result_label.pack(pady=20)
load_or_initialize_model()
window.mainloop()
