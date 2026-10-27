import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "plant_disease_model.keras"
)

classes = [
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight"
]

st.title("AI Crop Disease Prediction")

file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","png","jpeg"]
)

if file:
    img = Image.open(file)
    st.image(img)

    img = img.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)

    idx = np.argmax(pred)

    st.success(
        f"Disease: {classes[idx]}"
    )