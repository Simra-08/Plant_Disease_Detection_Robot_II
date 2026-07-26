import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("plant_model.h5")

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_healthy"
]

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[class_index], confidence

if __name__ == "__main__":
    result, conf = predict_image("test.jpg")
    print(f"Prediction: {result} ({conf*100:.1f}% confidence)")