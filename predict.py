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

img = Image.open("test_leaf.jpg")
img = img.resize((224,224))

img_array = np.array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)

idx = np.argmax(pred)

print("Prediction:", classes[idx])
print("Confidence:", float(np.max(pred))*100,"%")