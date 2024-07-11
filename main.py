import streamlit as st
import tensorflow as tf
import numpy as np
import gdown

# Function to download the model from Google Drive
def download_model_from_drive():
    url = 'https://drive.google.com/file/d/1z8hF_NpjzAQe_gBJco6pbXvfU5nFk9hH/view?usp=sharing'
    output = 'trained_model.h5'
    gdown.download(url, output, quiet=False)

# Function to load and predict using the model
def model_prediction(test_image):
    download_model_from_drive()
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")

# Main Page (Prediction Page)
st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
test_image = st.file_uploader("Choose an Image:")

if test_image is not None:
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        labels = [line.strip() for line in content]
        
        st.success("Model is predicting it's a {}".format(labels[result_index]))
