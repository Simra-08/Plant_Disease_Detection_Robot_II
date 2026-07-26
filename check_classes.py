from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(validation_split=0.2)
data = datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)
print(data.class_indices)