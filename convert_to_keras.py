import keras
from keras.layers import Dense, InputLayer

# ---------- patches ----------
_original_dense_from_config = Dense.from_config

@classmethod
def patched_dense_from_config(cls, config):
    config = dict(config)
    config.pop("quantization_config", None)
    return _original_dense_from_config.__func__(cls, config)

Dense.from_config = patched_dense_from_config

_original_input_from_config = InputLayer.from_config

@classmethod
def patched_input_from_config(cls, config):
    config = dict(config)
    config.pop("optional", None)

    if "batch_shape" in config and "shape" not in config and "input_shape" not in config:
        batch_shape = config.pop("batch_shape")
        if batch_shape is not None and len(batch_shape) > 1:
            config["shape"] = tuple(batch_shape[1:])
    else:
        config.pop("batch_shape", None)

    return _original_input_from_config.__func__(cls, config)

InputLayer.from_config = patched_input_from_config

# ---------- load ----------
model = keras.saving.load_model("mars_landmark_v2.h5", compile=False)
print("Loaded model successfully.")

# ---------- export ----------
model.export("mars_landmark_savedmodel")
print("Exported to mars_landmark_savedmodel/")