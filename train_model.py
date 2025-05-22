import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

# Parameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
IMAGE_SIZE = (224, 224)

# Dataset path (contains "with_mask" and "without_mask" subfolders)
dataset_dir = "data"

# Data generators with validation split
train_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2)

train_gen = train_aug.flow_from_directory(
    dataset_dir,
    target_size=IMAGE_SIZE,
    batch_size=BS,
    class_mode="categorical",
    subset="training")

val_gen = train_aug.flow_from_directory(
    dataset_dir,
    target_size=IMAGE_SIZE,
    batch_size=BS,
    class_mode="categorical",
    subset="validation")

# Save label binarizer (dictionary class_name: index)
label_map = train_gen.class_indices
with open("label_binarizer.pickle", "wb") as f:
    pickle.dump(label_map, f)

# Load MobileNetV2 base model (imagenet weights)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Build the head of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Combine base and head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("mask_detector.h5", monitor='val_loss', save_best_only=True)

# Train the model
H = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BS,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint])

# Evaluate model on validation data
val_gen.reset()
predIdxs = model.predict(val_gen, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

true_labels = val_gen.classes
labels = list(label_map.keys())

print(classification_report(true_labels, predIdxs, target_names=labels))

# Plot confusion matrix
cm = confusion_matrix(true_labels, predIdxs)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="Train Loss")
plt.plot(H.history["val_loss"], label="Val Loss")
plt.plot(H.history["accuracy"], label="Train Accuracy")
plt.plot(H.history["val_accuracy"], label="Val Accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()