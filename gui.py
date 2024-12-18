import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Function to load the trained model
def FacialExpressionModel(json_file, weights_file):
    try:
        with open(json_file, "r") as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the Haar Cascade and model
try:
    facec = cv2.CascadeClassifier("C:/Users/didor/Desktop/emotion detector/haarcascade_frontalface_default.xml")
    model = FacialExpressionModel("C:/Users/didor/Desktop/emotion detector/model_a.json", "C:/Users/didor/Desktop/emotion detector/model_weights.weights.h5")
    if model is None:
        raise ValueError("Model failed to load.")
except Exception as e:
    print(f"Error initializing: {e}")
    label1.configure(text="Initialization Error")
    label1.pack()
    top.mainloop()
    exit()

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to detect emotion
def Detect(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Invalid image path or unsupported file format.")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, 1.3, 5)

        if len(faces) == 0:
            label1.configure(foreground="#011638", text="No face detected")
            return

        for (x, y, w, h) in faces:
            fc = gray_image[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis] / 255.0  # Normalize the input
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
            label1.configure(foreground="#011638", text=f"Predicted Emotion: {pred}")
            print(f"Predicted Emotion: {pred}")
            return

    except Exception as e:
        print(f"Error in detection: {e}")
        label1.configure(foreground="#011638", text="Error in emotion detection")

# Function to show the detect button
def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return  # User canceled the file dialog

        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error in uploading image: {e}")
        label1.configure(foreground="#011638", text="Error in uploading image")

# GUI components
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
