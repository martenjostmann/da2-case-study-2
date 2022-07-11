from tensorflow import keras
from keras.applications.efficientnet_v2 import preprocess_input


def get_model():
  return keras.applications.EfficientNetV2S(weights="imagenet", include_top=False)


def get_preprocessor():
  return preprocess_input


def get_input_size():
  return (224, 224)
