from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
from datetime import datetime

app = Flask(__name__)
model = tf.keras.models.load_model("plant_model.h5")

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_healthy",
    "not_leaf"
]

# Treatment suggestions for each disease
treatments = {
    "Tomato_Early_blight": "Remove infected leaves. Apply a copper-based fungicide. Avoid overhead watering and improve air circulation between plants.",
    "Tomato_Late_blight": "Remove and destroy infected plants immediately to stop spread. Apply fungicide preventively during humid weather. Avoid wetting leaves when watering.",
    "Tomato_Leaf_Mold": "Improve ventilation in greenhouse/growing area. Reduce humidity. Apply fungicide if severe. Remove affected leaves.",
    "Tomato_healthy": "No action needed. Continue regular watering and monitoring.",
    "not_leaf": "Please point the camera at a tomato leaf and ensure good lighting."
}




latest_result = {"disease": None, "confidence": None, "treatment": None, "time": None}
latest_image_path = "latest.jpg"
history = []  # stores last few results

@app.route("/predict", methods=["POST"])
def predict():
    img_bytes = request.data
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    cv2.imwrite(latest_image_path, img)

    resized = cv2.resize(img, (224, 224))
    processed = preprocess_input(resized.astype(np.float32))
    processed = np.reshape(processed, (1, 224, 224, 3))

    prediction = model.predict(processed)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    disease = class_names[class_index]
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M:%S %p")

    if disease == "not_leaf":
        disease = "No leaf detected"
    latest_result["disease"] = disease
    latest_result["confidence"] = round(confidence * 100, 1)
    latest_result["treatment"] = treatments.get(disease, "No suggestion available.")
    latest_result["time"] = timestamp

    # Add to history (keep only last 8)
    history.insert(0, {
        "disease": disease,
        "confidence": round(confidence * 100, 1),
        "time": timestamp
    })
    if len(history) > 8:
        history.pop()

    return jsonify(latest_result)

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        disease=latest_result["disease"],
        confidence=latest_result["confidence"],
        treatment=latest_result["treatment"],
        time=latest_result["time"],
        history=history
    )

@app.route("/latest_image")
def latest_image():
    if os.path.exists(latest_image_path):
        return send_file(latest_image_path, mimetype="image/jpeg")
    return "", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)