"""Run the Mars landmark HEF on a Hailo-8 and report speed and accuracy."""
import csv
import os
import time
import numpy as np
from PIL import Image
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType,
)

HEF_PATH = "mars_model_float32.hef"
IMAGE_FOLDER = "data/map-proj-v3/"
TEST_CSV = "test.csv"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "other", "crater", "dark dune", "slope streak",
    "bright dune", "impact ejecta", "swiss cheese", "spider",
]

def preprocess_image(img_path):
    # Raw RGB uint8: normalization runs on the Hailo chip
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    return np.expand_dims(np.asarray(img, dtype=np.uint8), axis=0)

with open(TEST_CSV) as f:
    rows = list(csv.DictReader(f))

hef = HEF(HEF_PATH)
input_name = hef.get_input_vstream_infos()[0].name
output_name = hef.get_output_vstream_infos()[0].name

y_true, y_pred, inference_times = [], [], []

with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    with InferVStreams(network_group, input_params, output_params) as pipeline:
        with network_group.activate(network_group_params):
            # Warm-up
            first = preprocess_image(os.path.join(IMAGE_FOLDER, rows[0]["filename"]))
            pipeline.infer({input_name: first})

            for row in rows:
                x = preprocess_image(os.path.join(IMAGE_FOLDER, row["filename"]))

                start_time = time.perf_counter()
                output = pipeline.infer({input_name: x})
                end_time = time.perf_counter()

                preds = np.asarray(output[output_name]).reshape(-1)
                y_pred.append(int(np.argmax(preds)))
                y_true.append(int(row["label"]))
                inference_times.append(end_time - start_time)

y_true, y_pred = np.array(y_true), np.array(y_pred)
avg_time = np.mean(inference_times)

print("\n--- Performance Metrics ---")
print(f"Number of test images: {len(rows)}")
print(f"Average inference time: {avg_time * 1000:.3f} ms/image")
print(f"Average FPS: {1.0 / avg_time:.3f}")
print(f"Accuracy: {(y_true == y_pred).mean():.4f}")

try:
    from sklearn.metrics import classification_report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
except ImportError:
    print("(install scikit-learn for the per-class report)")