"""Build a calibration set of real Mars images for `hailo optimize`.

Images are saved as raw RGB 0-255 (no normalization). The normalization
is added to the model itself via the model script (mars_landmark.alls).
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_FOLDER = "data/map-proj-v3/"
TEST_CSV = "test.csv"
OUTPUT = "calib_set.npy"
IMG_SIZE = (224, 224)
NUM_IMAGES = 1024
SEED = 0

# Never calibrate on the test images
test_files = set(pd.read_csv(TEST_CSV)["filename"])
candidates = sorted(
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(".jpg") and f not in test_files
)

rng = np.random.default_rng(SEED)
chosen = rng.choice(candidates, size=min(NUM_IMAGES, len(candidates)), replace=False)

calib = np.zeros((len(chosen), IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.float32)
for i, name in enumerate(chosen):
    # Same loading as test_inference.py: RGB, nearest-neighbour resize
    img = Image.open(os.path.join(IMAGE_FOLDER, name)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.NEAREST)
    calib[i] = np.asarray(img, dtype=np.float32)

np.save(OUTPUT, calib)
print(f"Saved {OUTPUT}: shape={calib.shape}, range=[{calib.min():.0f}, {calib.max():.0f}]")