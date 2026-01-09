import numpy as np
import pandas as pd
from tflite_runtime.interpreter import Interpreter
from PIL import Image
from sklearn.metrics import classification_report
import serial
import os
import time

# --- CONFIG ---
TFLITE_MODEL = 'mars_model_quant.tflite'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']
SERIAL_PORT = '/dev/ttyHS1'
BAUD_RATE = 115200

# 1. Setup Serial Bridge
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.flushInput()
    HAS_SERIAL = True
    print(f"✅ Serial Link Active: {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Serial Link Failed: {e}")
    HAS_SERIAL = False

# 2. Setup Interpreter (Consistent with previous experiments)
interpreter = Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details['quantization']

# 3. Performance Tracking Variables (Consistent with previous experiments)
test_df = pd.read_csv(TEST_CSV)
y_true, y_pred, inference_times = [], [], []

print(f"🚀 Starting consistent benchmark on {len(test_df)} images...")
total_start_time = time.time()

# 4. Main Inference & Hardware Loop
try:
    for _, row in test_df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row['filename'])
        if not os.path.exists(img_path): continue

        # Optimized Preprocessing
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Quantize for MPU
        input_data = (img_array / input_scale) + input_zero_point
        input_data = np.expand_dims(input_data.astype(np.int8), axis=0)

        # 5. Time ONLY the inference (The Consistent Method)
        start_inference = time.time()
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        end_inference = time.time()
        
        # Store timing and classification results
        inference_times.append((end_inference - start_inference) * 1000)
        output_data = interpreter.get_tensor(output_details['index'])
        prediction = np.argmax(output_data)
        
        y_true.append(int(row['label']))
        y_pred.append(prediction)
        
        label = CLASS_NAMES[prediction]
        print(f"Targeting: {label:15} | Latency: {inference_times[-1]:.2f} ms")

        # 6. Hardware LED Trigger
        if HAS_SERIAL:
            if label == 'crater':
                ser.write(b'1')
            else:
                ser.write(b'0')
            ser.flush()

except KeyboardInterrupt:
    print("\n🛑 Mission interrupted. Calculating partial results...")

total_end_time = time.time()

# 7. Calculate Metrics (Consistent with previous experiments)
avg_time = np.mean(inference_times)
total_process_time = (total_end_time - total_start_time)

# 8. Final Report
report_header = "="*60 + "\nFINAL INTEGRATED PERFORMANCE REPORT\n" + "="*60
stats = (
    f"\nAverage Inference: {avg_time:.2f} ms\n"
    f"Throughput: {1000/avg_time:.2f} FPS\n"
    f"Total Test Time: {total_process_time:.2f} s\n"
)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

print(report_header + stats + report)

# Save to file for your records
with open("final_hardware_report.txt", "w") as f:
    f.write(report_header + stats + report)

if HAS_SERIAL: ser.close()