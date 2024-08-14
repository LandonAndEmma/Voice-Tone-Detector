import base64
import io
import os
import pickle
import tkinter as tk
import webbrowser
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

pygame.mixer.init()  # Initialize the pygame mixer for audio playback
canvas = None  # Initialize the canvas for displaying plots (used later in the code)
# Base64-encoded string representing the application icon
ICON_BASE64 = """iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAS1BMVEUAAADvjJS9vb1SSkLm5ube3t7Ozs7vKTrFxcXvvcWcnJzelJTvWmtaUlLeITHv7+/W1ta1tbWtra2lpaX3lJT3WmOtSkr3QkqcOkLJAJcIAAAAAXRSTlMAQObYZgAAAFFJREFUGNOljDcOgEAQA232EpfI4f8vRULao6CD6WZkGd+RAQTZq1spmcxUN6bsKwnFJzNO57M/Nm+tiKpzvlYX5hZSiPH+QaProGhYXgV/uAAQeQHIXWPCWwAAAABJRU5ErkJggg=="""


# Function to decode the base64 icon data and return a PhotoImage object
def get_icon_from_base64(base64_string):
    icon_data = base64.b64decode(base64_string)  # Decode the base64 string
    icon = Image.open(io.BytesIO(icon_data))  # Open the decoded data as an image
    return ImageTk.PhotoImage(icon)  # Convert the image to a PhotoImage object for use in Tkinter


# Function to extract audio features from a given audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
    # Compute the mean of Mel-frequency cepstral coefficients (MFCCs)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    # Compute the mean of the chroma feature
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    # Compute the mean of the Mel-scaled power spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    # Compute the mean of spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    # Concatenate all features into a single array and return it
    return np.hstack([mfccs, chroma, mel, contrast])


# Function to load an existing model and label encoder, or initialize a new one if none exist
def load_or_initialize_model():
    global model, le, X_train, y_train  # Declare global variables to store the model and training data
    model = SVC(kernel="linear")  # Initialize an SVM model with a linear kernel
    le = LabelEncoder()  # Initialize a label encoder for encoding tone labels
    # Check if the model and label encoder files exist
    if os.path.exists("tone_model.pkl") and os.path.exists("label_encoder.pkl"):
        # Load the saved model and label encoder
        with open("tone_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        # Check if the training data files exist
        if os.path.exists("X_train.npy") and os.path.exists("y_train.npy"):
            X_train = np.load("X_train.npy")  # Load training features
            y_train = np.load("y_train.npy")  # Load training labels
        else:
            X_train = np.array([])  # Initialize empty arrays if training data doesn't exist
            y_train = np.array([])
    else:
        X_train = np.array([])  # Initialize empty arrays if model and label encoder don't exist
        y_train = np.array([])
    # If there is no training data, train the model with predefined data
    if X_train.size == 0:
        print("No initial training data found. Training with predefined data.")
        initial_audio_files = ["neutral.wav", "happy.wav", "sad.wav", "mad.wav"]  # Predefined audio files
        initial_labels = ["neutral", "happy", "sad", "mad"]  # Corresponding labels
        data = []
        labels = []
        for audio, label in zip(initial_audio_files, initial_labels):
            if os.path.exists(audio):
                features = extract_features(audio)  # Extract features for each predefined audio file
                data.append(features)  # Add the features to the data list
                labels.append(label)  # Add the corresponding label to the labels list
            else:
                print(f"File {audio} not found!")
        if data:
            X_train = np.array(data)  # Convert the data list to a NumPy array
            y_train = np.array(labels)  # Convert the labels list to a NumPy array
            le.fit(y_train)  # Fit the label encoder with the training labels
            y_encoded = le.transform(y_train)  # Encode the labels
            model.fit(X_train, y_encoded)  # Train the SVM model with the training data
            # Save the trained model, label encoder, and training data to files
            with open("tone_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
            np.save("X_train.npy", X_train)
            np.save("y_train.npy", y_train)


# Function to predict the tone of a given audio file
def predict_tone(audio_path):
    global X_train, y_train  # Access global variables for the model and training data
    if os.path.exists(audio_path):  # Check if the audio file exists
        if X_train.size == 0:
            return "Model not trained yet. Please provide some initial data."
        features = extract_features(audio_path)  # Extract features from the audio file
        features = features.reshape(1, -1)  # Reshape features to match the input format of the model
        prediction = model.predict(features)  # Predict the tone using the trained model
        return le.inverse_transform(prediction)[0]  # Return the predicted tone label
    else:
        return "File not found!"


# Function to update the model with new training data and retrain it
def update_model(audio_path, correct_tone):
    global X_train, y_train  # Access global variables for the model and training data
    features = extract_features(audio_path)  # Extract features from the audio file
    # Add the new features to the existing training data
    X_train = (
        np.vstack([X_train, features]) if X_train.size else features.reshape(1, -1)
    )
    y_train = np.append(y_train, correct_tone)  # Append the correct label to the training labels
    le.fit(y_train)  # Refit the label encoder with the updated labels
    y_encoded = le.transform(y_train)  # Encode the updated labels
    model.fit(X_train, y_encoded)  # Retrain the model with the updated training data
    # Save the updated model, label encoder, and training data to files
    with open("tone_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)


# Function to plot the waveform of an audio file using matplotlib
def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path)  # Load the audio file
    fig, ax = plt.subplots(figsize=(8, 2))  # Create a figure and axis for the plot
    librosa.display.waveshow(y, sr=sr, ax=ax)  # Plot the waveform
    ax.set_title("Waveform")  # Set the title of the plot
    return fig  # Return the figure object


# Function to play an audio file using pygame
def play_audio(audio_path):
    pygame.mixer.music.load(audio_path)  # Load the audio file
    pygame.mixer.music.play()  # Play the audio file


# Function to open a file dialog and predict the tone of the selected audio file
def open_file():
    global canvas  # Access the global canvas variable
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav")])  # Open file dialog to select a WAV file
    if file_path:
        tone = predict_tone(file_path)  # Predict the tone of the selected file
        result_label.config(text=f"Predicted Tone: {tone}")  # Display the predicted tone
        if canvas:
            canvas.get_tk_widget().destroy()  # Destroy the previous canvas if it exists
        fig = plot_waveform(file_path)  # Plot the waveform of the selected file
        canvas = FigureCanvasTkAgg(fig, master=window)  # Embed the plot into the Tkinter window
        canvas.draw()  # Draw the canvas
        canvas.get_tk_widget().pack(pady=20)  # Pack the canvas widget with padding
        play_audio(file_path)  # Play the selected audio file
        response = messagebox.askquestion(
            "Confirm Prediction", f"Is the prediction '{tone}' correct?"
        )  # Ask the user to confirm the predicted tone
        if response == "no":
            correct_tone = simpledialog.askstring(
                "Correct Tone", "Please enter the correct tone:"
            )  # Ask the user for the correct tone if the prediction was wrong
            if correct_tone:
                update_model(file_path, correct_tone)  # Update the model with the correct tone
                messagebox.showinfo(
                    "Model Updated", "The model has been updated with the correct tone."
                )  # Inform the user that the model was updated
            else:
                messagebox.showwarning(
                    "Input Error",
                    "No correct tone provided. The model was not updated.",
                )  # Warn the user if no correct tone was provided
        else:
            messagebox.showinfo(
                "Confirmation",
                "The prediction was confirmed as correct. No updates were made to the model.",
            )  # Inform the user if the prediction was confirmed as correct


# Function to display a help message
def open_help():
    messagebox.showinfo("Help",
                        "This program allows you to use and train an AI to figure out the tone of vocals\n\n"
                        "Code: Landon & Emma\n")


# Function to open the project's GitHub repository in a web browser
def open_repository():
    webbrowser.open_new("https://github.com/LandonAndEmma/Voice-Tone-Detector")


# Function to handle the window close event
def on_closing():
    # Perform any cleanup here if necessary (e.g., stopping audio playback)
    pygame.mixer.music.stop()  # Stop any playing audio
    window.destroy()  # Close the window and stop the Tkinter main loop


# Create the main Tkinter window
window = tk.Tk()
window.title("Voice Tone Detector")  # Set the window title
icon = get_icon_from_base64(ICON_BASE64)  # Get the application icon from the base64 string
window.iconphoto(False, icon)  # Set the window icon
window.geometry("800x301")  # Set the window size
window.minsize(275, 301)  # Set the minimum window size
window.maxsize(800, 301)  # Set the maximum window size
window.resizable(True, False)  # Make the window resizable horizontally but not vertically
window.attributes('-fullscreen', False)  # Disable fullscreen mode
window.protocol("WM_DELETE_WINDOW", on_closing)  # Bind the window close event to the on_closing function
# Create a menu bar and add file and help menus
menubar = tk.Menu(window)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)  # Add an "Open" command to the file menu
menubar.add_cascade(label="File", menu=file_menu)  # Add the file menu to the menu bar
help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="Help", command=open_help)  # Add a "Help" command to the help menu
help_menu.add_command(label="Repository", command=open_repository)  # Add a "Repository" command to the help menu
menubar.add_cascade(label="Help", menu=help_menu)  # Add the help menu to the menu bar
window.config(menu=menubar)  # Set the menu bar for the window
result_label = tk.Label(window, text="Predicted Tone: None")  # Create a label to display the predicted tone
result_label.pack(pady=20)  # Pack the label with padding
load_or_initialize_model()  # Load or initialize the model when the program starts
window.mainloop()  # Start the Tkinter main loop to run the application
