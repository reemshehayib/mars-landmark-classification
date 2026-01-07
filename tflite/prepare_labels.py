from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMG_DIR = DATA / "map-proj-v3"

labels_path = DATA / "labels-map-proj-v3.txt"
classmap_path = DATA / "landmarks_map-proj-v3_classmap.csv"

# ---- classmap: force no header (some files have no header row) ----
cm = pd.read_csv(classmap_path, header=None, names=["class_id", "class_name"])
cm["class_id"] = cm["class_id"].astype(int)
cm = cm.sort_values("class_id").reset_index(drop=True)
print("Classmap rows:", len(cm))
print(cm.to_string(index=False))

# ---- labels: keep only images that exist on disk ----
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
        p = IMG_DIR / fname
        if p.exists():
            rows.append((fname, str(p), y))

df = pd.DataFrame(rows, columns=["filename", "path", "y"])

out_labels = DATA / "labels_found.csv"
df.to_csv(out_labels, index=False)

print("\nWrote:", out_labels)
print("Rows:", len(df))
print("Class distribution:")
print(df["y"].value_counts().sort_index().to_string())

print("\nUnique class IDs present:", sorted(df["y"].unique().tolist()))
