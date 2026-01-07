import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import os

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
IMAGE_FOLDER = 'data/map-proj-v3/' # Ensure this matches your directory
NUM_CLASSES = 8

# 1. Load Data Maps & Dynamic Weights
train_df = pd.read_csv('train.csv', dtype={'label': str})
val_df = pd.read_csv('val.csv', dtype={'label': str})

# Dynamic weight calculation
y_train = train_df['label'].astype(int).values
raw_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weights_dict = dict(zip(np.unique(y_train), raw_weights))
print(f"Dynamic Weights: {weights_dict}")

# 2. Advanced Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=applications.mobilenet_v2.preprocess_input,
    rotation_range=90,      # Mars landmarks can be seen from any angle
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,         # Different satellite altitudes
    horizontal_flip=True,
    vertical_flip=True,     # Crucial for top-down satellite imagery
    fill_mode='nearest'
)

train_gen = datagen.flow_from_dataframe(
    train_df, directory=IMAGE_FOLDER, x_col='filename', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_gen = datagen.flow_from_dataframe(
    val_df, directory=IMAGE_FOLDER, x_col='filename', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False
)

# 3. Build & Phase 1 (Warmup)
base_model = applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Phase 1: Training the Head ---")
model.fit(train_gen, validation_data=val_gen, epochs=5, class_weight=weights_dict)

# 4. Phase 2: Fine-Tuning (The "Accuracy Booster")
print("\n--- Phase 2: Fine-Tuning deeper layers ---")
base_model.trainable = True
# Freeze the bottom 100 layers, only tune the top ones (specialized features)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Use a MUCH lower learning rate for fine-tuning
model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Stop early if the model stops getting better
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=15, 
    class_weight=weights_dict,
    callbacks=[early_stop]
)

# 5. Save final model
model.save('mars_landmark_v2.h5')
print("Model Saved!")