import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('retina_model.h5')


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title('Diabetic Retinopathy Detection using CNN:')
st.write('Upload an image of the retina to get the prediction.')

uploaded_image = st.file_uploader('Choose an image...', type='png')
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f'Prediction: {predicted_class}')
