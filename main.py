import os
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# ---------------------------------------------------------
# 1. Image Data Generators
# ---------------------------------------------------------
print("\n--- Starting Data Loading ---")
base_path = r"D:\traffic_signB"
train_dir = os.path.join(base_path, "train")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load test data directly using DataFrame
test_image_dir = os.path.join(base_path, "test", "images")
test_csv_file = os.path.join(base_path, "test", "GT-final_test.csv")
df_test = pd.read_csv(test_csv_file, sep=';')
# Keras flow_from_dataframe requires the label to be a string if class_mode='categorical'
df_test['ClassId'] = df_test['ClassId'].astype(str).str.zfill(5)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=test_image_dir,
    x_col='Filename',
    y_col='ClassId',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print("Datasets loaded successfully!")

# ---------------------------------------------------------
# 3. Build CNN Model
# ---------------------------------------------------------
print("\n--- Building and Training Model ---")
model = Sequential([
    Input(shape=(64,64,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // 32, validation_data=val_generator, validation_steps=val_generator.samples // 32, epochs=15)

# Evaluate model
print("\n--- Evaluating Model ---")
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("traffic_sign_model.h5")
print("Model saved as traffic_sign_model.h5")

# ---------------------------------------------------------
# 4. Plot Results and Confusion Matrix
# ---------------------------------------------------------
# Plot training results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training & Validation Accuracy")
plt.show()

# Predict on test set
y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification report (precision, recall, f1-score per class)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))
