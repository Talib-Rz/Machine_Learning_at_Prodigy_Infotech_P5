import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os

def download_model_from_drive():
    url = 'https://drive.google.com/file/d/1z8hF_NpjzAQe_gBJco6pbXvfU5nFk9hH/view?usp=sharing'
    output = 'trained_model.h5'
    response = requests.get(url)
    with open(output, 'wb') as f:
        f.write(response.content)

def model_prediction(test_image):
    download_model_from_drive()
    model_path = "trained_model.h5"
    
    try:
        model = tf.keras.models.load_model(model_path)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)
        
        # Reading Labels
        labels_file = "labels.txt"
        if os.path.isfile(labels_file):
            with open(labels_file) as f:
                content = f.readlines()
            labels = [line.strip() for line in content]
            return result_index, labels[result_index]
        else:
            st.error(f"File '{labels_file}' not found.")
            return result_index, "Unknown"
    
    except OSError as e:
        st.error(f"Error loading model: {e}")
        return -1, "Error"
    except Exception as e:
        st.error(f"Exception during prediction: {e}")
        return -1, "Error"

# Streamlit app code
st.sidebar.title("Hi! I'm Talib")
st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")

st.header("Model Prediction")
test_image = st.file_uploader("Choose an Image:")
if st.button("Show Image"):
    st.image(test_image, use_column_width=True)

# Predict button
if st.button("Predict"):
    st.write("Our Prediction")
    result_index, label = model_prediction(test_image)
    if result_index != -1:
        st.success(f"Model is predicting it's a {label}")
