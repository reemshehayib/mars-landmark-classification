import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
# If your images are in the same folder as this script, keep './'
# If they are in a folder named 'images', change it to './images/'
IMAGE_FOLDER = 'data/map-proj-v3/' 
LABELS_FILE = 'data/labels-map-proj-v3.txt'

def prepare_dataset():
    # 1. Load the master label list
    if not os.path.exists(LABELS_FILE):
        print(f"Error: {LABELS_FILE} not found!")
        return

    print(f"Reading {LABELS_FILE}...")
    df_master = pd.read_csv(LABELS_FILE, sep=" ", names=["filename", "label"], dtype={"label": str})
    
    # 2. Get list of files actually present on disk
    print(f"Scanning {IMAGE_FOLDER} for actual images...")
    files_on_disk = set(os.listdir(IMAGE_FOLDER))
    
    # 3. FILTER: Keep only the rows where the image file actually exists
    df = df_master[df_master['filename'].isin(files_on_disk)].copy()
    
    if len(df) == 0:
        print("Error: No images found on disk. Check your IMAGE_FOLDER path.")
        return

    print(f"Success! Matched {len(df)} images from disk with the labels file.")

    # 4. Stratified Split (80% Train/Val, 20% Test)
    # 'stratify' ensures each set has the same % of craters, dunes, etc.
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df['label'], random_state=42
    )

    # 5. Split Train/Val (12.5% of the 80% = 10% of total for validation)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, stratify=train_val_df['label'], random_state=42
    )

    # 6. Calculate Class Weights
    # This tells the GPU to pay more attention to rare classes
    unique_labels = np.unique(train_df['label'])
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_df['label']
    )
    
    # Create a dictionary for Keras
    weights_dict = {int(label): weight for label, weight in zip(unique_labels, weights)}

    # 7. Save the manifests to CSV
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print("\n✅ Setup Complete!")
    print(f"Total available images: {len(df)}")
    print(f"---")
    print(f"Training set:   {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set:       {len(test_df)} images (Hidden from model until the end)")
    print("\n--- Class Weights (Copy these for the next step) ---")
    print(weights_dict)

if __name__ == "__main__":
    prepare_dataset()