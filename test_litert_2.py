import numpy as np
import pandas as pd
import os
import time
from PIL import Image
from sklearn.metrics import classification_report

# 1. Setup LiteRT / TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- CONFIG ---
TFLITE_MODEL = 'mars_model_quant.tflite'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']
INPUT_SIZE = (224, 224)

interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Check quantization
is_quantized = input_details['dtype'] == np.int8
if is_quantized:
    input_scale, input_zero_point = input_details['quantization']

# 2. Performance Tracking
test_df = pd.read_csv(TEST_CSV)
y_true, y_pred, inference_times = [], [], []

print(f"🚀 Starting benchmark on {len(test_df)} images...")

# 3. Bulk Inference Loop
total_start_time = time.time()

for _, row in test_df.iterrows():
    img_path = os.path.join(IMAGE_FOLDER, row['filename'])
    if not os.path.exists(img_path): continue

    # Preprocess (Lightweight PIL method)
    img = Image.open(img_path).convert('RGB').resize(INPUT_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if is_quantized:
        input_data = (img_array / input_scale) + input_zero_point
        input_data = np.expand_dims(input_data, axis=0).astype(np.int8)
    else:
        input_data = np.expand_dims(img_array, axis=0)

    # Time ONLY the inference
    t0 = time.time()
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    t1 = time.time()
    
    inference_times.append((t1 - t0) * 1000) # Convert to ms
    output_data = interpreter.get_tensor(output_details['index'])
    y_true.append(int(row['label']))
    y_pred.append(np.argmax(output_data))

total_end_time = time.time()

# 4. Calculate Latency & Throughput (FPS)
avg_inference_ms = np.mean(inference_times)
# FPS is 1 / (time in seconds), so 1000 / (time in ms)
fps = 1000 / avg_inference_ms 
total_script_time = total_end_time - total_start_time

# 5. Final Report
report_header = "="*60 + "\nMARS LANDMARK PERFORMANCE REPORT\n" + "="*60
stats = (
    f"\n[LATENCY & SPEED]\n"
    f"Avg Inference Time: {avg_inference_ms:.2f} ms\n"
    f"Inference Speed:    {fps:.2f} FPS\n"
    f"Total Images:       {len(y_true)}\n"
    f"Total Script Time:  {total_script_time:.2f} s\n"
)

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
full_output = f"{report_header}\n{stats}\n[CLASSIFICATION METRICS]\n{report}"

print(full_output)

with open("final_deployment_report.txt", "w") as f:
    f.write(full_output)