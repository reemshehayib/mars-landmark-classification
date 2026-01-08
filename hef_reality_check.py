import numpy as np
import cv2
import pandas as pd
import os
from hailo_platform import (VDevice, HEF, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, 
                            FormatType, HailoStreamInterface)

# --- 1. CONFIGURATION ---
HEF_PATH = 'mars_landmark.hef'
CSV_PATH = 'test.csv'
DATA_DIR = 'data/map-proj-v3/' 

# --- 2. DATA LOADING ---
df = pd.read_csv(CSV_PATH)
filename = df['filename'].iloc[0]
true_label = df['label'].iloc[0]
full_path = os.path.join(DATA_DIR, filename)

if not os.path.exists(full_path):
    print(f"❌ Error: Could not find image at {full_path}")
    exit()

# Load image and resize to model dimensions
img = cv2.imread(full_path)
img_resized = cv2.resize(img, (224, 224))

# CRITICAL: Convert to uint8 (0-255 range). 
# This ensures the buffer size is exactly 150,528 bytes (224*224*3).
input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
input_data = np.ascontiguousarray(input_data)

# --- 3. HAILO INFERENCE ---
with VDevice() as target:
    hef = HEF(HEF_PATH)
    
    # Configure the hardware (PCIe for Pi 5)
    conf_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, conf_params)[0]
    
    # Set input to UINT8 and output to FLOAT32 (HailoRT handles de-quantization for us)
    input_vstreams_params = InputVStreamParams.make_from_network_group(
        network_group, format_type=FormatType.UINT8
    )
    output_vstreams_params = OutputVStreamParams.make_from_network_group(
        network_group, format_type=FormatType.FLOAT32
    )
    
    # Generate parameters needed for activation
    network_group_params = network_group.create_params()

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name

        data_to_send = {input_name: input_data}
        
        with network_group.activate(network_group_params):
            # Perform inference
            result = infer_pipeline.infer(data_to_send)
            
            # Argmax gives us the index of the predicted class
            pred_class = np.argmax(result[output_name])
            
            print("="*40)
            print(f"✅ SUCCESS: Inference Ran Successfully")
            print(f"🎯 Image: {filename}")
            print(f"🎯 True Label: {true_label} | 🚀 Hailo Pred: {pred_class}")
            print("="*40)