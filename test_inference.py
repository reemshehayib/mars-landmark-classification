import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

# --- CONFIG ---
MODEL_DIR = "mars_landmark_savedmodel"
IMAGE_FOLDER = "data/map-proj-v3/"
TEST_CSV = "test.csv"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "other",
    "crater",
    "dark dune",
    "slope streak",
    "bright dune",
    "impact ejecta",
    "swiss cheese",
    "spider"
]

print("GPUs:", tf.config.list_physical_devices("GPU"))

# 1. Load SavedModel
loaded = tf.saved_model.load(MODEL_DIR)

print("Available signatures:", list(loaded.signatures.keys()))

# Usually Keras export creates 'serving_default'
infer = loaded.signatures["serving_default"]

# 2. Load test data
test_df = pd.read_csv(TEST_CSV)

y_true = []
y_pred = []
inference_times = []

# 3. Helper to preprocess one image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

# 4. Warm-up
first_img_path = os.path.join(IMAGE_FOLDER, test_df.iloc[0]["filename"])
warmup_input = preprocess_image(first_img_path)
warmup_output = infer(warmup_input)

# 5. Inference loop
for _, row in test_df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row["filename"])
    x = preprocess_image(img_path)

    start_time = time.perf_counter()
    output = infer(x)
    end_time = time.perf_counter()

    # output is a dict; take first tensor
    preds = list(output.values())[0].numpy()

    pred_class = int(np.argmax(preds, axis=1)[0])
    true_class = int(row["label"])

    y_pred.append(pred_class)
    y_true.append(true_class)
    inference_times.append(end_time - start_time)

# 6. Metrics
avg_time = np.mean(inference_times)
avg_time_ms = avg_time * 1000
avg_fps = 1.0 / avg_time if avg_time > 0 else 0

print("\n--- Performance Metrics ---")
print(f"Number of test images: {len(test_df)}")
print(f"Average inference time: {avg_time_ms:.3f} ms/image")
print(f"Average FPS: {avg_fps:.3f}")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))