from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMG_DIR = DATA / "map-proj-v3"

labels_path = DATA / "labels-map-proj-v3.txt"
classmap_path = DATA / "landmarks_map-proj-v3_classmap.csv"

# classmap (force: no header)
cm = pd.read_csv(classmap_path, header=None, names=["class_id", "class_name"])
cm["class_id"] = cm["class_id"].astype(int)

print("Classes:", cm["class_id"].nunique())
print(cm.sort_values("class_id").to_string(index=False))

# labels
rows = []
with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        fname, y = parts[0], int(parts[1])
        rows.append((fname, y))

df = pd.DataFrame(rows, columns=["filename", "y"])
df["path"] = df["filename"].apply(lambda x: IMG_DIR / x)
exists = df["path"].apply(lambda p: p.exists())
print("\nLabel rows:", len(df))
print("Images found:", int(exists.sum()), "missing:", int((~exists).sum()))

# label distribution
dist = df[exists]["y"].value_counts().sort_index()
print("\nClass distribution (found images):")
print(dist.to_string())

# quick peek at image count on disk
num_imgs = sum(1 for _ in IMG_DIR.glob("*.jpg"))
print("\nImages in folder:", num_imgs)
