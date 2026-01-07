import os
os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

MODEL_FP16 = "mars_mnv3_fp16.tflite"
MODEL_INT8_IOFLOAT = "mars_mnv3_int8_iofloat.tflite"  # int8 weights, float I/O

VAL_CSV = "data/val.csv"
CLASSMAP_CSV = "data/landmarks_map-proj-v3_classmap.csv"

IMG_SIZE = 224
NUM_THREADS = 4
N_SAMPLES = 10
SEED = 0


def load_classmap():
    cm = pd.read_csv(CLASSMAP_CSV, header=None, names=["id", "name"])
    cm["id"] = cm["id"].astype(int)
    cm["name"] = cm["name"].astype(str)
    return dict(zip(cm["id"].tolist(), cm["name"].tolist()))


def preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img).astype(np.float32)
    # MobileNetV3 preprocess_input => [-1, 1]
    x = (x / 127.5) - 1.0
    return x[None, ...].astype(np.float32)


def make_interpreter(model_path: str):
    itp = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=NUM_THREADS,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    itp.allocate_tensors()
    return itp



def run_one(itp, x: np.ndarray):
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]

    if inp["dtype"] != np.float32:
        raise RuntimeError(f"Expected float32 input, got {inp['dtype']}")

    itp.set_tensor(inp["index"], x)
    itp.invoke()
    y = itp.get_tensor(out["index"])[0]

    # dequantize output if needed
    if out["dtype"] == np.int8:
        scale, zero = out["quantization"]
        y = scale * (y.astype(np.float32) - zero)

    pred = int(np.argmax(y))
    conf = float(np.max(y))
    return pred, conf


def main():
    cm = load_classmap()
    df = pd.read_csv(VAL_CSV).sample(n=min(N_SAMPLES, 10_000), random_state=SEED)
    itp_fp16 = make_interpreter(MODEL_FP16)
    itp_i8   = make_interpreter(MODEL_INT8_IOFLOAT)


    print(f"Testing on {len(df)} validation images:\n")

    for _, r in df.iterrows():
        path = str(r["path"])
        y_true = int(r["y"])

        x = preprocess(path)

        p16, c16 = run_one(itp_fp16, x)
        p8, c8 = run_one(itp_i8, x)

        fname = path.split("/")[-1]
        print(fname)
        print(f"  true: {y_true} ({cm.get(y_true,'?')})")
        print(f"  fp16: {p16} ({cm.get(p16,'?')})  conf={c16:.4f}")
        print(f"  int8_iofloat(portable): {p8} ({cm.get(p8,'?')})  conf={c8:.4f}")
        print()


if __name__ == "__main__":
    main()
