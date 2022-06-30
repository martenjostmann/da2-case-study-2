from tensorflow import keras

def simple_model(input_shape = (256, 256, 3)):
  model = keras.Sequential()
  model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform',input_shape=input_shape))
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
  model.add(keras.layers.BatchNormalization(center=True, scale=True))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dense(64, activation="relu"))
  model.add(keras.layers.Dense(32, activation="relu"))
  model.add(keras.layers.Dense(5, activation='softmax'))
  
  return model