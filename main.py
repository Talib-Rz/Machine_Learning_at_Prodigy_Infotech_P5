import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load the model once when the app starts
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "trained_model.h5"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to make a prediction
def model_prediction(image, model):
    try:
        # Process the image
        image = image.resize((64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        
        # Make prediction
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)
        
        # Read labels
        labels_file = "labels.txt"
        if os.path.isfile(labels_file):
            with open(labels_file) as f:
                content = f.readlines()
            labels = [line.strip() for line in content]
            return result_index, labels[result_index]
        else:
            st.error(f"File '{labels_file}' not found.")
            return result_index, "Unknown"
    
    except Exception as e:
        st.error(f"Exception during prediction: {e}")
        return -1, "Error"

# Streamlit app code
st.sidebar.title("Hi! I'm Talib")
st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")

# Load the model
model = load_model()

if model is not None:
    st.header("Model Prediction")
    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index, label = model_prediction(image, model)
            if result_index != -1:
                st.success(f"Model is predicting it's a {label}")
else:
    st.error("Model could not be loaded.")
