import numpy as np
import cv2
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from hailo_platform import (VDevice, HEF, InferVStreams, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, 
                            FormatType, HailoStreamInterface)

# --- CONFIGURATION ---
HEF_PATH = 'mars_landmark.hef'
CSV_PATH = 'test.csv'
DATA_DIR = 'data/map-proj-v3/' 

# Exact mapping based on your provided list
CLASS_NAMES = [
    'Other',          # 0
    'Crater',         # 1
    'Dark Dune',      # 2
    'Slope Streak',   # 3
    'Bright Dune',    # 4
    'Impact Ejecta',  # 5
    'Swiss Cheese',   # 6
    'Spider'          # 7
]

def run_full_evaluation():
    df = pd.read_csv(CSV_PATH)
    y_true = []
    y_pred = []
    latencies = []
    
    print(f"🚀 Starting evaluation on {len(df)} images...")

    with VDevice() as target:
        hef = HEF(HEF_PATH)
        conf_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, conf_params)[0]
        
        # Using Float32 for best precision alignment with the training data
        input_vparams = InputVStreamParams.make_from_network_group(network_group, format_type=FormatType.FLOAT32)
        output_vparams = OutputVStreamParams.make_from_network_group(network_group, format_type=FormatType.FLOAT32)
        network_group_params = network_group.create_params()

        with InferVStreams(network_group, input_vparams, output_vparams) as infer_pipeline:
            input_name = hef.get_input_vstream_infos()[0].name
            output_name = hef.get_output_vstream_infos()[0].name

            with network_group.activate(network_group_params):
                for index, row in df.iterrows():
                    full_path = os.path.join(DATA_DIR, row['filename'])
                    img_bgr = cv2.imread(full_path)
                    if img_bgr is None: continue
                    
                    # Preprocessing: BGR to RGB and Normalization
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    input_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
                    input_tensor = np.ascontiguousarray(input_tensor)
                    
                    # Inference
                    start_inf = time.time()
                    result = infer_pipeline.infer({input_name: input_tensor})
                    end_inf = time.time()
                    
                    latencies.append(end_inf - start_inf)
                    
                    # Record Results
                    y_true.append(int(row['label']))
                    y_pred.append(np.argmax(result[output_name]))

                    if index % 100 == 0 and index > 0:
                        print(f"✅ Processed {index}/{len(df)} images...")

    # --- METRICS CALCULATION ---
    avg_latency = (sum(latencies) / len(latencies)) * 1000
    acc_val = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f"📊 PERFORMANCE SUMMARY")
    print(f"Avg Latency:    {avg_latency:.2f} ms")
    print(f"Final Accuracy: {acc_val * 100:.2f}%")
    print("="*60)

    # --- THE NEW PART: PRECISION, RECALL, F1 ---
    print("\n📈 DETAILED CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("="*60)

    return y_true, y_pred, acc_val

# --- EXECUTION ---
y_actual, y_predicted, final_acc = run_full_evaluation()

# Generate and Save Confusion Matrix Plot
cm = confusion_matrix(y_actual, y_predicted)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
plt.title(f'Mars Landmark Evaluation\nAccuracy: {final_acc*100:.2f}%')
plt.tight_layout()
plt.savefig('mars_confusion_matrix.png')

print("\n✨ Evaluation complete. Report printed and matrix saved as 'mars_confusion_matrix.png'")