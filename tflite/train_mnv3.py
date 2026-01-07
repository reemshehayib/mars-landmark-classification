from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).parent
DATA = ROOT / "data"

TRAIN_CSV = DATA / "train.csv"
VAL_CSV   = DATA / "val.csv"

IMG_SIZE = 224
BATCH = 64
NUM_CLASSES = 8
SEED = 42

# ---------- load splits ----------
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print("Train:", len(train_df), "Val:", len(val_df))
print("Train class counts:\n", train_df["y"].value_counts().sort_index().to_string())
print("Val class counts:\n", val_df["y"].value_counts().sort_index().to_string())

# ---------- class weights (for imbalance) ----------
counts = train_df["y"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).values.astype(np.float32)
weights = np.sqrt(counts.sum() / (counts + 1e-6))
weights = weights / weights.mean()
class_weight = {i: float(weights[i]) for i in range(NUM_CLASSES)}
print("\nClass weights:", class_weight)

AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, label, training: bool):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # force RGB for pretrained weights
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.keras.applications.mobilenet_v3.preprocess_input(tf.cast(img, tf.float32))


    # light aug only (you already have rotated/flipped/brightness variants on disk)
    if training:
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_brightness(img, 0.05)

    return img, tf.cast(label, tf.int32)

def make_ds(df: pd.DataFrame, training: bool):
    paths = df["path"].astype(str).values
    labels = df["y"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: load_and_preprocess(p, y, training),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

ds_train = make_ds(train_df, training=True)
ds_val   = make_ds(val_df, training=False)

# ---------- model ----------
base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)

x = tf.keras.layers.Dropout(0.1)(base.output)
out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(base.input, out)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best.keras", monitor="val_accuracy", save_best_only=True),
]

# Phase 1: train head
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print("\n[Phase 1] Training head...")
model.fit(ds_train, validation_data=ds_val, epochs=15, callbacks=callbacks)


for images, labels in ds_val.take(1):
    preds = model(images, training=False)
    print("Pred class counts:", np.bincount(tf.argmax(preds, axis=1).numpy(), minlength=8))
    print("True class counts:", np.bincount(labels.numpy(), minlength=8))


# Phase 2: fine-tune last part only (keeps it stable)
# Unfreeze base, but freeze early layers for stability
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print("\n[Phase 2] Fine-tuning...")
model.fit(ds_train, validation_data=ds_val, epochs=10, callbacks=callbacks)

for images, labels in ds_val.take(1):
    preds = model(images, training=False)
    print("Pred class counts:", np.bincount(tf.argmax(preds, axis=1).numpy(), minlength=8))
    print("True class counts:", np.bincount(labels.numpy(), minlength=8))

model.export("mars_mnv3_savedmodel")
print("Saved: mars_mnv3_savedmodel")
