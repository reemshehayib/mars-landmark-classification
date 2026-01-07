import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os
import time

# --- CONFIG ---
TFLITE_MODEL = 'mars_model_quant.tflite'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']

# 1. Setup Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details['quantization']

# 2. Performance Tracking Variables
test_df = pd.read_csv(TEST_CSV)
y_true = []
y_pred = []
inference_times = []

print(f"🚀 Starting benchmark on {len(test_df)} images...")

# 3. Bulk Inference Loop
total_start_time = time.time()

for _, row in test_df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row['filename'])
    if not os.path.exists(img_path): continue

    # Preprocess
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    input_data = (np.expand_dims(img_array, axis=0) / input_scale) + input_zero_point
    input_data = input_data.astype(np.int8)

    # Time ONLY the inference (interpreter.invoke)
    start_inference = time.time()
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    end_inference = time.time()
    
    # Store results
    inference_times.append((end_inference - start_inference) * 1000) # Convert to ms
    output_data = interpreter.get_tensor(output_details['index'])
    y_true.append(int(row['label']))
    y_pred.append(np.argmax(output_data))

total_end_time = time.time()

# 4. Calculate Metrics
avg_time = np.mean(inference_times)
total_inference_only = np.sum(inference_times)
total_process_time = (total_end_time - total_start_time) # Includes loading/processing

# 5. Final Report
report_header = "="*60 + "\nMARS LANDMARK PERFORMANCE REPORT\n" + "="*60
performance_stats = (
    f"\n[LATENCY METRICS]\n"
    f"Average Inference Time: {avg_time:.2f} ms per image\n"
    f"Total Pure Inference:   {total_inference_only/1000:.2f} seconds\n"
    f"Total Script Duration:  {total_process_time:.2f} seconds\n"
    f"Throughput:             {1000/avg_time:.2f} frames per second (FPS)\n"
)

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

print(report_header)
print(performance_stats)
print("[CLASSIFICATION METRICS]")
print(report)

# Save to file
with open("final_deployment_report.txt", "w") as f:
    f.write(report_header + performance_stats + "\n[CLASSIFICATION METRICS]\n" + report)