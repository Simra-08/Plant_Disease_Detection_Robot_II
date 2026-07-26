import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

model = tf.keras.models.load_model("plant_model_best.h5")

preds = model.predict(val_data, steps=len(val_data), verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes[:len(y_pred)]

class_labels = list(val_data.class_indices.keys())
all_labels = list(range(len(class_labels)))  # ensures all 5 classes are included even if some missing from this batch

print("\nClass indices:", val_data.class_indices)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=all_labels, target_names=class_labels, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=all_labels))