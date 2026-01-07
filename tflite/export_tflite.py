import tensorflow as tf
import pandas as pd
import numpy as np

SAVEDMODEL_DIR = "mars_mnv3_savedmodel"
TRAIN_CSV = "data/train.csv"

IMG_SIZE = 224

# -------- FP16 (easy + fast on Pi/UNO Q) --------
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()

with open("mars_mnv3_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)

print("Wrote mars_mnv3_fp16.tflite")

# -------- INT8 (smaller + fastest) --------
df = pd.read_csv(TRAIN_CSV).sample(n=min(300, len(pd.read_csv(TRAIN_CSV))), random_state=42)

def rep_dataset():
    for p in df["path"].astype(str).values:
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.keras.applications.mobilenet_v3.preprocess_input(tf.cast(img, tf.float32))
        img = tf.expand_dims(img, 0)
        yield [img]

# ---------- INT8 weights + float32 I/O (Fix 2) ----------
converter = tf.lite.TFLiteConverter.from_saved_model(SAVEDMODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_int8_iofloat = converter.convert()
with open("mars_mnv3_int8_iofloat.tflite", "wb") as f:
    f.write(tflite_int8_iofloat)

print("Wrote mars_mnv3_int8_iofloat.tflite")

