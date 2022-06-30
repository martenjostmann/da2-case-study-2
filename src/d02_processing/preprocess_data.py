import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
import numpy as np

def class_weights(dataset):
  # Compute class weights
  labels = [np.argmax(y.numpy()) for x, y in dataset.unbatch()]
  classes = np.unique(labels)
  weights = compute_class_weight('balanced', classes=classes,y=labels)
  weights = {i:weight for i, weight in enumerate(weights)}

  return weights


def normalize_image(image):
    return image/255

def split_data(dataset):
    
    # split data into train and val set
    train_dataset = dataset.take(int(len(dataset)*0.8))
    val_dataset = dataset.skip(int(len(dataset)*0.8))
    
    return train_dataset, val_dataset

def one_hot_encode(dataset):
    # one hot encode labels
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, 5)))

    return dataset

def shuffle_data(dataset):
    # shuffle data
    dataset = dataset.shuffle(len(dataset), seed=21,reshuffle_each_iteration=False)
    
    return dataset

def zoom_image(image, zoom_factor=2):
  h = image.shape[0]

  w = image.shape[1]
  image = tf.image.resize(image, (tf.cast(h*zoom_factor, tf.int32), tf.cast(w*zoom_factor, tf.int32)))

  trim_top = ((image.shape[0] - h) // 2)
  trim_left = ((image.shape[1] - w) // 2)
  image = image[trim_top:trim_top + h, trim_left:trim_left + w]

  return image


def preprocess_train(dataset, one_hot:bool=True, size=None):
    
    #normalize images
    dataset = dataset.map(lambda x,y : (normalize_image(x),y))
    
    dataset = dataset.map(lambda x,y : (zoom_image(x),y))

    # resize images
    if size is not None:
      dataset = dataset.map(lambda x, y: (tf.image.resize(x, size),y))

    if one_hot:
      dataset = one_hot_encode(dataset)
    
    #split into background and relevant class
    dataset_background = dataset.take(3110)
    dataset_relevant = dataset.skip(3110)
    
    #first shuffle of background and relevant class
    dataset_background = shuffle_data(dataset_background)
    dataset_relevant = shuffle_data(dataset_relevant)
    
    # split data of background and relevant class
    dataset_background_train, dataset_background_val = split_data(dataset_background)
    dataset_relevant_train, dataset_relevant_val = split_data(dataset_relevant)
    
    
    #Data augmentation
    data_augmentation = tf.keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomRotation(0.4),
    ])
    dataset_relevant_train = dataset_relevant_train.map(lambda x,y: (data_augmentation(x),y))
    dataset_relevant_val = dataset_relevant_val.map(lambda x,y: (data_augmentation(x),y))

    #Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    dataset_relevant_train = dataset_relevant_train.cache().prefetch(buffer_size=AUTOTUNE).repeat(20)
    dataset_relevant_val = dataset_relevant_val.cache().prefetch(buffer_size=AUTOTUNE).repeat(20)
    
    dataset_background_train = dataset_background_train.cache().prefetch(buffer_size=AUTOTUNE)
    dataset_background_val = dataset_background_val.cache().prefetch(buffer_size=AUTOTUNE)
    
    train_dataset = dataset_relevant_train.concatenate(dataset_background_train)
    val_dataset = dataset_relevant_val.concatenate(dataset_background_val)
    
    train_dataset = shuffle_data(train_dataset)
    val_dataset = shuffle_data(val_dataset)
    
    return train_dataset.batch(32), val_dataset.batch(1)

def preprocess_patches(dataset, size=None):
    #normalize images
    dataset = dataset.map(lambda x : normalize_image(x))

    dataset = dataset.map(lambda x : zoom_image(x))

    # resize images
    if size is not None:
      dataset = dataset.map(lambda x: tf.image.resize(x, size))

    AUTOTUNE = tf.data.AUTOTUNE
    dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset
