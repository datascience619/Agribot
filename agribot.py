import numpy as np
import pandas as pd
import tkinter as tk
import os
from tkinter import filedialog, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from nltk.chat.util import Chat, reflections

#from t1 import train_generator

# Sample soil data and crop recommendations
soil_data = {
    'pH': [6.5, 7.0, 5.5, 6.0],
    'nitrogen': [20, 25, 15, 18],
    'phosphorus': [10, 12, 8, 9],
    'potassium': [15, 20, 10, 12],
    'crop': ['Wheat', 'Rice', 'Maize', 'Barley']
}

# Convert soil data to DataFrame
soil_df = pd.DataFrame(soil_data)

# Train crop recommendation model
X = soil_df[['pH', 'nitrogen', 'phosphorus', 'potassium']]
y = soil_df['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier()
crop_model.fit(X_train, y_train)

# Load pre-trained disease prediction model
# You can replace this with the actual path to your model
# Load the dataset correctly
data = pd.read_csv('Crop_recommendation.csv')
print(data.head())  # Display the first few rows of the dataset
# Define a simple chatbot using NLTK
agricultural_responses = [
    (r'hi|hello|hey', ['Hello! How can I assist you with your farming queries today?']),
    (r'what crop should I grow?', ['Please provide your soil pH, nitrogen, phosphorus, and potassium levels.']),
    (r'my soil has pH (.), nitrogen (.), phosphorus (.), potassium (.)',
     ['Based on your soil parameters, I recommend growing: {}']),
    (r'how do I treat (.*) disease?', [
        'For treating {}, you can try the following remedies:To treat apple leaf diseases, focus on prevention through sanitation, pruning, and potentially fungicide applications, while also selecting disease-resistant varieties where possible']),
    (r'suggest some schemes of government for farmers?', [
        'These are following schemes of government for farmers: Modified Interest Subvention Scheme (MISS), Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)']),
    (r'quit', ['Thank you for using the agricultural chatbot. Have a great day!']),
]
chatbot = Chat(agricultural_responses, reflections)

# Initialize the uploaded image variable
uploaded_image = None


# Functions
def recommend_crop():
    try:
        ph = float(ph_entry.get())
        nitrogen = float(nitrogen_entry.get())
        phosphorus = float(phosphorus_entry.get())
        potassium = float(potassium_entry.get())
        prediction = crop_model.predict([[ph, nitrogen, phosphorus, potassium]])
        result_label.config(text=f"Recommended Crop: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values")


def upload_image():
    global uploaded_image
    file_path = filedialog.askopenfilename()
    if file_path:
        uploaded_image = file_path  # Store the uploaded image path
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        result_label.config(text="Image uploaded successfully! Now click 'Predict Disease' to predict the disease.")


# def predict_disease(disease_model=None):
#     if uploaded_image:
#         img = image.load_img(uploaded_image, target_size=(224, 224))
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         prediction = disease_model.predict(img_array)
#         disease_class = np.argmax(prediction, axis=1)
#         result_label.config(text=f"Predicted Disease: {disease_class[0]}")
#     else:
#         messagebox.showerror("No Image", "Please upload an image first using the 'Upload Image' button.")
#
#
#         data = pd.read_csv(r'\Users\Admin\Downloads\archive (4)\PlantVillage')
#         print(data.head())
#
#         def predict_disease(train_generator=None):
#             if uploaded_image:
#                 img = image.load_img(uploaded_image, target_size=(224, 224))
#                 img_array = image.img_to_array(img) / 255.0
#                 img_array = np.expand_dims(img_array, axis=0)
#
#                 prediction = disease_model.predict(img_array)
#                 disease_class = np.argmax(prediction, axis=1)
#
#                 class_labels = list(train_generator.class_indices.keys())  # Get class names
#                 result_label.config(text=f"Predicted Disease: {class_labels[disease_class[0]]}")
#             else:
#                 messagebox.showerror("No Image", "Please upload an image first using the 'Upload Image' button.")


# Load model only if it exists
model_path = 'C:/Users/Nafees/PycharmProjects/PythonProject/plant_disease_model.h5'

if os.path.exists(model_path):
    disease_model = load_model(model_path)
else:
    print("Error: Model file not found!")
    disease_model = None  # Avoid using NoneType object




def predict_disease(train_generator=None):
    global disease_model  # Ensure the model is accessible

    if disease_model is None:
        messagebox.showerror("Error", "Disease model is not loaded. Please check the model file.")
        return

    if uploaded_image:
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = disease_model.predict(img_array)
        disease_class = np.argmax(prediction, axis=1)

        class_labels = [
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy'
        ]
        # ← use your actual classes
        # Ensure correct labels
  # Ensure correct labels
        result_label.config(text=f"Predicted Disease: {class_labels[disease_class[0]]}")
    else:
        messagebox.showerror("No Image", "Please upload an image first using the 'Upload Image' button.")


def chat():
    user_input = chat_entry.get()
    if user_input.lower() == 'quit':
        result_label.config(text="Chatbot: Goodbye!")
    else:
        response = chatbot.respond(user_input)
        result_label.config(text=f"Chatbot: {response}")


# GUI Setup
root = tk.Tk()
root.title("Agricultural Assistant")
root.geometry("500x600")
root.config(bg="#f4f4f9")  # Light background color for the app

# Title Label
title_label = tk.Label(root, text="Agricultural Assistant", font=("Arial", 18, "bold"), bg="#f4f4f9")
title_label.pack(pady=10)

# Frame for Soil Parameters
soil_frame = tk.Frame(root, bg="#f4f4f9")
soil_frame.pack(pady=20)

tk.Label(soil_frame, text="Enter Soil Parameters", font=("Arial", 14, "bold"), bg="#f4f4f9").pack(pady=5)

tk.Label(soil_frame, text="pH:", font=("Arial", 12), bg="#f4f4f9").pack(anchor="w", padx=10)
ph_entry = tk.Entry(soil_frame, font=("Arial", 12))
ph_entry.pack(pady=5, padx=10, fill='x')

tk.Label(soil_frame, text="Nitrogen:", font=("Arial", 12), bg="#f4f4f9").pack(anchor="w", padx=10)
nitrogen_entry = tk.Entry(soil_frame, font=("Arial", 12))
nitrogen_entry.pack(pady=5, padx=10, fill='x')

tk.Label(soil_frame, text="Phosphorus:", font=("Arial", 12), bg="#f4f4f9").pack(anchor="w", padx=10)
phosphorus_entry = tk.Entry(soil_frame, font=("Arial", 12))
phosphorus_entry.pack(pady=5, padx=10, fill='x')

tk.Label(soil_frame, text="Potassium:", font=("Arial", 12), bg="#f4f4f9").pack(anchor="w", padx=10)
potassium_entry = tk.Entry(soil_frame, font=("Arial", 12))
potassium_entry.pack(pady=5, padx=10, fill='x')

tk.Button(soil_frame, text="Recommend Crop", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white",
          command=recommend_crop).pack(pady=10, padx=10, fill='x')

# Frame for Disease Prediction
disease_frame = tk.Frame(root, bg="#f4f4f9")
disease_frame.pack(pady=20)

# First Button: Upload Image for Disease Prediction
upload_button = tk.Button(disease_frame, text="Upload Image", font=("Arial", 12, "bold"), bg="#FF5722", fg="white",
                          command=upload_image)
upload_button.pack(pady=10, padx=10, fill='x')

# Second Button: Predict Disease After Image Upload
predict_button = tk.Button(disease_frame, text="Predict Disease", font=("Arial", 12, "bold"), bg="#FF9800", fg="white",
                           command=predict_disease)
predict_button.pack(pady=10, padx=10, fill='x')

# Frame for Chatbot
chat_frame = tk.Frame(root, bg="#f4f4f9")
chat_frame.pack(pady=20)

tk.Label(chat_frame, text="Chat with Bot", font=("Arial", 14, "bold"), bg="#f4f4f9").pack(pady=5)
chat_entry = tk.Entry(chat_frame, font=("Arial", 12))
chat_entry.pack(pady=5, padx=10, fill='x')

tk.Button(chat_frame, text="Send", font=("Arial", 12, "bold"), bg="#2196F3", fg="white", command=chat).pack(pady=10,
                                                                                                            padx=10,
                                                                                                            fill='x')

# Result Label for displaying recommendations or responses
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#f4f4f9")
result_label.pack(pady=20)

# Run the GUI
root.mainloop()
