import numpy as np
import pandas as pd
import cv2
import os
from hailo_sdk_client import ClientRunner

# --- 1. CONFIGURATION ---
model_name = "mars_landmark"
base_path = 'data/map-proj-v3/'
test_df = pd.read_csv('test.csv')

# --- 2. LOAD 1024 IMAGES FOR CALIBRATION ---
calibration_images = []
# We use more images to ensure the quantization 'scales' are accurate
for fname in test_df['filename'].iloc[:1024]: 
    img_path = os.path.join(base_path, fname)
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    if img is not None:
        # Swap BGR to RGB to match training
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        calibration_images.append(img.astype(np.uint8))

calib_dataset = np.array(calibration_images)
print(f"✅ Loaded {len(calib_dataset)} images for calibration.")

# --- 3. RUN THE HAILO FLOW ---
runner = ClientRunner(hw_arch='hailo8l')
runner.translate_tf_model('mars_model_float.tflite', model_name)

# --- 4. THE ACCURACY FIX: MODEL SCRIPT ---
# We ONLY use normalization for classification models
# This maps 0-255 UINT8 pixels to the 0.0-1.0 FLOAT range
model_script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"

runner.load_model_script(model_script)

# 5. Optimize (Quantization)
# This step uses the 1024 images to find the best 8-bit weights
runner.optimize(calib_dataset)

# 6. Compile
hef = runner.compile()
with open(f"{model_name}.hef", "wb") as f:
    f.write(hef)

print("🚀 HEF Compiled Successfully with Normalization!")