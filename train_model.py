import tensorflow as tf
from  keras.src.legacy.preprocessing.image import ImageDataGenerator


import os

# Paths
TRAIN_DIR = "data/train"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

# Data generator (split train into train+validation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 80% train, 20% validation
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save the model
model.save(os.path.join(MODEL_DIR, "asl_model.h5"))
print("✅ Model saved successfully at models/asl_model.h5")

