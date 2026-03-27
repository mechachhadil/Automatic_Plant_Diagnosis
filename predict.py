"""
This script allows us to test our trained CNN model on a single image.

Workflow:
1. Load trained model
2. Load and preprocess image
3. Run prediction
4. Display predicted class + confidence score
"""

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import pickle

# Import config values
from config import MODEL_FILE, IMAGE_SIZE

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
# Load the saved CNN model (.keras file)
model = tf.keras.models.load_model(MODEL_FILE)

# ==========================================================
# LOAD CLASS NAMES
# ==========================================================
# IMPORTANT:
# You must define the same class order used during training
# Ideally, save it during training → here we define manually

with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# ==========================================================
# IMAGE PREPROCESSING FUNCTION
# ==========================================================
def preprocess_image(img_path):
    """
    Load and preprocess an image to match model input.

    Steps:
    - Load image
    - Resize to model input size
    - Convert to array
    - Expand dimensions (batch format)
    - Apply MobileNetV2 preprocessing
    """

    # Load image from disk
    img = image.load_img(img_path, target_size=IMAGE_SIZE)

    # Convert image to numpy array
    img_array = image.img_to_array(img)

    # Add batch dimension → shape becomes (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply same preprocessing used during training
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    return img_array

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def predict_image(img_path):
    """
    Predict the class of a given image.

    Returns:
    - predicted class name
    - confidence score
    """

    # Preprocess image
    processed_image = preprocess_image(img_path)

    # Run prediction
    predictions = model.predict(processed_image)

    # Get index of highest probability
    predicted_index = np.argmax(predictions[0])

    # Get confidence score
    confidence = predictions[0][predicted_index]

    # Map index → class name
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":

    # Ask user to input image path
    img_path = input("Enter image path: ")
    print("RAW INPUT:", repr(img_path))
    print("EXISTS:", os.path.exists(img_path))

    # Check if file exists
    if not os.path.exists(img_path):
        print("Error: Image not found!")
        exit()

    # Predict
    predicted_class, confidence = predict_image(img_path)

    # Display results
    print("\nPrediction Results:")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")