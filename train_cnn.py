import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import joblib

print("Loading Spectrogram Images...")
df = pd.read_pickle("cnn_audio_features.pkl")

X = np.array(df['feature'].tolist())
y = np.array(df['class'].tolist())

# Reshape data to fit a CNN: (samples, height, width, color_channels)
# Spectrograms are 1 color channel (grayscale)
img_height, img_width = X[0].shape
X = X.reshape(X.shape[0], img_height, img_width, 1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Building CNN Architecture...")
model = Sequential([
    # Layer 1: Feature Detection
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Layer 2: Deep Feature Detection
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Layer 3: Complex Pattern Detection
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.3),

    # Transition from Image to Classification
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    
    # Output Layer
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training CNN Vision Model...")
# CNNs are heavier, so we train for 30 epochs instead of 50
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

print("\nSaving CNN Model...")
model.save("cnn_language_model.h5")
joblib.dump(le, "cnn_label_encoder.pkl")
print("Complete! Ready for deployment.")