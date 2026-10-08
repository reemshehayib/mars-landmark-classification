"""Compare float vs quantized accuracy in the Hailo emulator (no device needed)."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from hailo_sdk_client import ClientRunner, InferenceContext

HAR = "mars_landmark_quant.har"
IMAGE_FOLDER = "data/map-proj-v3/"
TEST_CSV = "test.csv"
IMG_SIZE = (224, 224)

test_df = pd.read_csv(TEST_CSV)
labels = test_df["label"].to_numpy()

# Raw RGB 0-255: normalization is now inside the model
images = np.zeros((len(test_df), IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
for i, name in enumerate(test_df["filename"]):
    img = Image.open(os.path.join(IMAGE_FOLDER, name)).convert("RGB")
    images[i] = np.asarray(img.resize(IMG_SIZE, Image.NEAREST), dtype=np.float32)

runner = ClientRunner(har=HAR)

def accuracy(context_type):
    with runner.infer_context(context_type) as ctx:
        out = runner.infer(ctx, images, batch_size=16)
    if isinstance(out, (list, tuple)):
        out = out[0]
    preds = np.asarray(out).reshape(len(images), -1).argmax(axis=1)
    return (preds == labels).mean(), preds

float_acc, _ = accuracy(InferenceContext.SDK_FP_OPTIMIZED)
quant_acc, quant_preds = accuracy(InferenceContext.SDK_QUANTIZED)

print(f"\nTest images:        {len(labels)}")
print(f"Float accuracy:     {float_acc:.4f}")
print(f"Quantized accuracy: {quant_acc:.4f}")
print(f"Drop:               {float_acc - quant_acc:.4f}")