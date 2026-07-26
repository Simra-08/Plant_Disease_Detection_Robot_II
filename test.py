import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("plant_model.h5")

# Class names (IMPORTANT – must match your folders)
class_names = [
    "Tomato___Early_blight",
    "Tomato___Healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold"
]

# Load image
img = cv2.imread("test.jpg")   # put your test image name here
img = cv2.resize(img, (224,224))
img = preprocess_input(img)
img = np.reshape(img, (1, 224, 224, 3))

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)

print("Prediction:", class_names[class_index])