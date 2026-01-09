import numpy as np
import pandas as pd
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import serial  # Native way to talk to the STM32
import os
import time

# --- CONFIG ---
TFLITE_MODEL = 'mars_model_quant.tflite'
IMAGE_FOLDER = 'data/map-proj-v3/'
TEST_CSV = 'test.csv'
CLASS_NAMES = ['other', 'crater', 'dark dune', 'slope streak', 'bright dune', 'impact ejecta', 'swiss cheese', 'spider']

# 1. Setup Serial Bridge (The "Nervous System")
try:
    # On Arduino UNO Q, the internal serial is usually /dev/ttyMSM1
    ser = serial.Serial('/dev/ttyHS1', 115200, timeout=1)
    HAS_SERIAL = True
    print("✅ Serial bridge to Arduino connected.")
except Exception as e:
    print(f"⚠️ Serial bridge failed: {e}. Running in Simulation Mode.")
    HAS_SERIAL = False

# 2. Setup TFLite Interpreter
interpreter = Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale, input_zero_point = input_details['quantization']

test_df = pd.read_csv(TEST_CSV)
print("🚀 Rover AI Initialized. Processing Mars Terrain...")

try:
    for _, row in test_df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row['filename'])
        if not os.path.exists(img_path): continue

        # 3. Preprocessing
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        input_data = (img_array / input_scale) + input_zero_point
        input_data = np.expand_dims(input_data.astype(np.int8), axis=0)

        # 4. Inference
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        prediction = np.argmax(interpreter.get_tensor(output_details['index']))
        label = CLASS_NAMES[prediction]

        print(f"Detected: {label}")

        # 5. Hardware Action
        if HAS_SERIAL:
            if label == 'crater':
                ser.write(b'1')  # Sending as a single byte
                ser.flush()      # Force the data out of the buffer immediately
            else:
                ser.write(b'0')
                ser.flush()

        time.sleep(0.1)

except KeyboardInterrupt:
    if HAS_SERIAL: ser.close()
    print("\n🛑 Mission Aborted.")