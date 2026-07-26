import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import Counter
import numpy as np

# ================= SETTINGS =================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30            # early stopping will likely stop before this

# ================= DATA AUGMENTATION =================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    horizontal_flip=True
)

# ================= TRAIN DATA =================
train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# ================= VALIDATION DATA =================
val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ================= CLASS INFO =================
print("\nClass indices:")
print(train_data.class_indices)

# ================= CLASS WEIGHTS =================
labels = train_data.classes
counter = Counter(labels)
max_count = float(max(counter.values()))
class_weights = {cls: max_count / count for cls, count in counter.items()}
print("\nClass weights:")
print(class_weights)

# ================= BASE MODEL =================
base_model = EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True
for layer in base_model.layers[:-30]:   # unfreeze a few more layers than before
    layer.trainable = False

# ================= CUSTOM HEAD =================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ================= COMPILE =================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================= CALLBACKS =================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "plant_model_best.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ================= TRAIN =================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ================= FINAL SAVE =================
model.save("plant_model.h5")
print("\nFinal model saved as plant_model.h5")
print("Best model (by val_accuracy) saved as plant_model_best.h5")

# ================= EVALUATION =================
print("\n" + "="*50)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*50)
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# ================= DETAILED REPORT =================
from sklearn.metrics import classification_report, confusion_matrix

val_data.reset()
preds = model.predict(val_data, steps=len(val_data), verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes[:len(y_pred)]

class_labels = list(val_data.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nTraining complete!")