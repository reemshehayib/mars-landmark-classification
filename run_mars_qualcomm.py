import numpy as np
import pandas as pd
from tflite_runtime.interpreter import Interpreter
from PIL import Image
from sklearn.metrics import classification_report
import os
import time

# --- CONFIG ---
TFLITE_MODEL = 'mars_model_quant.tflite'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']

# 1. Setup Interpreter (Optimized for ARM64 MPU)
interpreter = Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Pre-fetch quantization parameters
input_scale, input_zero_point = input_details['quantization']

# 2. Performance Tracking Variables
test_df = pd.read_csv(TEST_CSV)
y_true, y_pred, inference_times = [], [], []

print(f"🚀 Qualcomm MPU: Starting benchmark on {len(test_df)} images...")

# 3. Inference Loop
total_start_time = time.time()

for _, row in test_df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row['filename'])
    if not os.path.exists(img_path): continue

    # 4. Optimized Preprocessing with Pillow
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Quantize for the Qualcomm MPU
    input_data = (img_array / input_scale) + input_zero_point
    input_data = np.expand_dims(input_data.astype(np.int8), axis=0)

    # 5. Time ONLY the inference
    start_inference = time.time()
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    end_inference = time.time()
    
    # Store results
    inference_times.append((end_inference - start_inference) * 1000)
    output_data = interpreter.get_tensor(output_details['index'])
    y_true.append(int(row['label']))
    y_pred.append(np.argmax(output_data))

total_end_time = time.time()

# 6. Calculate Metrics
avg_time = np.mean(inference_times)
total_process_time = (total_end_time - total_start_time)

# 7. Final Report
report_header = "="*60 + "\nMARS QUALCOMM PERFORMANCE REPORT\n" + "="*60
stats = (
    f"\nAverage Inference: {avg_time:.2f} ms\n"
    f"Throughput: {1000/avg_time:.2f} FPS\n"
)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

print(report_header + stats + report)

with open("qualcomm_report.txt", "w") as f:
    f.write(report_header + stats + report)