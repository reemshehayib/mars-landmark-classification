import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# --- CONFIG ---
MODEL_PATH = 'mars_landmark_v2.h5'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
# Labels from your CSV
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']

# 1. Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load test data
test_df = pd.read_csv(TEST_CSV)
samples = test_df.sample(5) # Pick 5 random images

plt.figure(figsize=(15, 5))

for i, (index, row) in enumerate(samples.iterrows()):
    img_path = os.path.join(IMAGE_FOLDER, row['filename'])
    
    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds)
    actual_class = int(row['label'])
    
    # Plot
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    title_color = 'green' if pred_class == actual_class else 'red'
    plt.title(f"Pred: {CLASS_NAMES[pred_class]}\nActual: {CLASS_NAMES[actual_class]}", color=title_color)
    plt.axis('off')

plt.tight_layout()
plt.show()