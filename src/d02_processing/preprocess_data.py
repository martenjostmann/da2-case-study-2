import tensorflow as tf
from tensorflow import keras

def normalize_image(image):
    return image/255

def preprocess(train_dataset, val_dataset):
    
    #normalize images
    train_dataset = train_dataset.map(lambda x,y : (normalize_image(x),y))
    val_dataset = val_dataset.map(lambda x,y : (normalize_image(x),y))
    
    #Data augmentation
    data_augmentation = tf.keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    train_dataset = train_dataset.map(lambda x,y: (data_augmentation(x),y))

    #Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE).repeat(2)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset.batch(32), val_dataset.batch(1)