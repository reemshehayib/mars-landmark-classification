import tensorflow as tf
import numpy as np
import pandas as pd
import os

# --- CONFIG ---
MODEL_PATH = 'mars_landmark_v2.h5' # Ensure this matches your saved file
IMAGE_FOLDER = 'data/map-proj-v3/'
VAL_CSV = 'val.csv'

# 1. Representative Dataset Generator
# This tells the converter what "typical" Mars landmark data looks like
def representative_data_gen():
    # We use 100 random images from your validation set to calibrate weights
    val_df = pd.read_csv(VAL_CSV).sample(100)
    for _, row in val_df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row['filename'])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Apply the exact same preprocessing used in training
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        yield [img_array]

# 2. Conversion Process
print("🚀 Starting LiteRT INT8 Conversion...")
model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimizations to reduce size
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Enforce full integer quantization (Required for Hailo and Qualcomm DSPs)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# 3. Save the Edge Model
with open('mars_model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Success! 'mars_model_quant.tflite' is ready for your hardware.")