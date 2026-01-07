from pathlib import Path
import re
import random
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "labels_found.csv")

AUG_SUFFIXES = ["-r90", "-r180", "-r270", "-fh", "-fv", "-brt"]

def base_id(filename: str) -> str:
    stem = Path(filename).stem
    for s in AUG_SUFFIXES:
        stem = stem.replace(s, "")
    return stem

df["group"] = df["filename"].apply(base_id)

groups = df["group"].unique().tolist()
random.seed(42)
random.shuffle(groups)

val_frac = 0.15
n_val = int(len(groups) * val_frac)
val_groups = set(groups[:n_val])

train_df = df[~df["group"].isin(val_groups)].reset_index(drop=True)
val_df   = df[df["group"].isin(val_groups)].reset_index(drop=True)

# Save splits
out_train = DATA / "train.csv"
out_val   = DATA / "val.csv"
train_df.to_csv(out_train, index=False)
val_df.to_csv(out_val, index=False)

print("Total images:", len(df))
print("Unique groups:", len(groups))
print("Train images:", len(train_df), "Val images:", len(val_df))
print("Train groups:", train_df["group"].nunique(), "Val groups:", val_df["group"].nunique())

print("\nTrain class distribution:")
print(train_df["y"].value_counts().sort_index().to_string())

print("\nVal class distribution:")
print(val_df["y"].value_counts().sort_index().to_string())

# sanity: ensure no group overlap
overlap = set(train_df["group"].unique()).intersection(set(val_df["group"].unique()))
print("\nGroup overlap:", len(overlap))
