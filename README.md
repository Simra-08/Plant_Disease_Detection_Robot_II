# 🌱 Plant Disease Detection Rover

An AI-powered smart agriculture system that combines **robotics, IoT, and deep learning** to detect diseases in tomato plants. The rover can be controlled wirelessly through a web browser while streaming live video from an ESP32-CAM. Captured images are analyzed using an EfficientNetB0-based deep learning model to identify common tomato leaf diseases in real time.

---

## 📌 Overview

Plant diseases significantly affect crop yield and quality. Traditional disease detection relies on manual inspection, which can be time-consuming and requires agricultural expertise. This project automates the monitoring process by integrating a mobile rover, wireless camera streaming, and an AI-based disease classification model.

The system enables users to remotely navigate the rover through agricultural fields, capture live images of plant leaves, and receive instant disease predictions.

---

## ✨ Features

- 🚗 Wireless browser-controlled rover
- 📷 Live video streaming using ESP32-CAM
- 🧠 AI-powered tomato disease detection
- 🌐 Browser-based control interface
- ⚡ Real-time image prediction
- 🍅 Detects multiple tomato leaf diseases
- 🔄 Deep learning model trained using transfer learning

---

## 🛠 Hardware Used

- ESP32 Development Board
- ESP32-CAM Module
- L298N Motor Driver
- 2 DC Motors
- Robot Chassis
- 2 × 18650 Batteries
- Jumper Wires
- USB Cable

---

## 💻 Software & Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- Flask
- Arduino IDE
- HTML
- CSS
- ESP32 Wi-Fi Library

---

## 🧠 AI Model

The disease detection model is built using **EfficientNetB0** with transfer learning.

### Dataset

The model was trained using the **PlantVillage Tomato Leaf Dataset** obtained from Kaggle.

### Diseases Detected

- Tomato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold

### Training Techniques

- Transfer Learning
- Data Augmentation
- Image Rotation
- Zoom
- Horizontal Flip
- Brightness Adjustment
- Validation Split
- Confidence Threshold Filtering

---

## ⚙ Working

1. The ESP32 rover connects to Wi-Fi.
2. Users access the rover through a browser interface.
3. The rover moves wirelessly in all directions.
4. ESP32-CAM streams live video.
5. Python captures frames from the camera stream.
6. The trained EfficientNetB0 model classifies the leaf image.
7. The predicted disease is displayed to the user.

---

## 📂 Project Structure

```
Plant-Disease-Detection-Rover
│
├── Arduino/
│   ├── Rover_Code.ino
│   └── ESP32_CAM_Code.ino
│
├── AI/
│   ├── train.py
│   ├── capture.py
│   ├── server.py
│   ├── predict.py
│   └── plant_model.h5
│
├── Images/
│
├── README.md
└── requirements.txt
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Plant-Disease-Detection-Rover.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Run live detection

```bash
python capture.py
```

---

## 🎮 Rover Controls

The rover is controlled through a browser.

Available controls:

- Forward
- Backward
- Left
- Right
- Stop

The ESP32 hosts a simple web interface that communicates with the motor driver.

---

## 📊 Experimental Results

The developed system successfully demonstrated:

- Stable wireless rover movement
- Live ESP32-CAM video streaming
- Real-time tomato disease prediction
- Improved prediction accuracy after data augmentation
- Reduced false predictions using confidence thresholding

The system performs best under good lighting conditions with clear leaf images.

---

## 🔮 Future Enhancements

- Support multiple crop species
- Autonomous navigation
- GPS integration
- Cloud-based monitoring
- Mobile application
- Disease severity estimation
- Automatic pesticide recommendation

---

## 👨‍💻 Team

Developed as a smart agriculture project integrating:

- Robotics
- Internet of Things (IoT)
- Computer Vision
- Deep Learning

---

## 📸 Demo

(Add screenshots of)

- Rover
- ESP32-CAM live stream
- Browser control interface
- Disease prediction output

---

## 📄 License

This project is intended for educational and research purposes.
