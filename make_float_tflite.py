import tensorflow as tf
from keras.src.layers.core.dense import Dense

# This "CustomDense" ignores the extra quantization tag that's causing the crash
class CustomDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# Load the model while telling Keras to use our CustomDense instead of the standard one
try:
    model = tf.keras.models.load_model(
        'mars_landmark_model.h5', 
        custom_objects={'Dense': CustomDense}
    )
    print("✅ Model loaded successfully!")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('mars_model_float.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✅ Created mars_model_float.tflite")

except Exception as e:
    print(f"❌ Still failing: {e}")