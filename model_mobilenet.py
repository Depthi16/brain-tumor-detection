import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

data = []
labels = []

categories = ["tumor", "no_tumor"]
path = "dataset"

for category in categories:
    folder = os.path.join(path, category)
    label = categories.index(category)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (128, 128))
        data.append(image)
        labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

model.save("mobilenet_model.h5")