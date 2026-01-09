import numpy as np
import pandas as pd
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import serial
import os
import time

# --- CONFIGURATION ---
TFLITE_MODEL = 'mars_model_quant.tflite'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']
SERIAL_PORT = '/dev/ttyHS1' # Correct high-speed port for UNO Q
BAUD_RATE = 115200

# 1. Initialize Serial Connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    # Clear any junk data in the buffer
    ser.flushInput()
    ser.flushOutput()
    HAS_SERIAL = True
    print(f"✅ Connected to STM32 at {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Serial connection failed: {e}")
    HAS_SERIAL = False

# 2. Setup TFLite Interpreter
interpreter = Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details['quantization']

# 3. Load Test Data
test_df = pd.read_csv(TEST_CSV)
print("🚀 Rover AI Online. Starting Autonomy Mode...")

try:
    for _, row in test_df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row['filename'])
        if not os.path.exists(img_path): continue

        # Preprocessing
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        input_data = (img_array / input_scale) + input_zero_point
        input_data = np.expand_dims(input_data.astype(np.int8), axis=0)

        # Inference
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        prediction = np.argmax(interpreter.get_tensor(output_details['index']))
        label = CLASS_NAMES[prediction]

        print(f"Targeting: {label:15}")

        # 4. Send Command to Hardware
        if HAS_SERIAL:
            if label == 'crater':
                ser.write(b'1\n') # Signal '1' with newline
            else:
                ser.write(b'0\n') # Signal '0' with newline
            ser.flush() # Ensure it leaves the Python buffer immediately

        # Small delay to sync with LED visual persistence
        time.sleep(0.2)

except KeyboardInterrupt:
    if HAS_SERIAL: ser.close()
    print("\n🛑 Mission Aborted by Pilot.")