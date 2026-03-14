import tensorflow as tf

saved_model_dir = "/home/rshehayi/reem_tests/mars-landmark-classification/mars_landmark_savedmodel"
output_tflite = "/home/rshehayi/reem_tests/mars-landmark-classification/mars_model_float32.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Keep it FLOAT. Do not add quantization settings.
tflite_model = converter.convert()

with open(output_tflite, "wb") as f:
    f.write(tflite_model)

print(f"Saved float32 TFLite model to: {output_tflite}")