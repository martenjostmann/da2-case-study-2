from sklearn.preprocessing import LabelEncoder
import numpy as np
from random import random

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras

from src.d03_model import transfer_learning


# We have 3110 background images. Therefore, the predefined split.
# Not used in our current approach anymore since len(dataset) cannot be reliably used with datasets.
def oversample(dataset, split_index=3110):
    dataset_bg = dataset.take(split_index)
    dataset_rel = dataset.skip(split_index)

    mult = len(dataset_bg) / len(dataset_rel)

    if mult < 1:
        return dataset

    dataset_rel = dataset_rel.repeat(int(mult))

    return dataset_bg.concatenate(dataset_rel)


def one_hot_encode(dataset):
    """
    One Hot encode the class labels, for instance class label 3 will result in [0, 0, 0, 1, 0]

    Parameters
    ----------
    dataset: tf.data.Dataset
      Dataset with two elements [data, label]

    Returns
    -------
    returns: tf.data.Dataset
      Dataset with one hot encoded class labels
    """

    return dataset.map(lambda x, y: (x, tf.one_hot(y, 5)))


def split(dataset, split_factor=0.3, dataset_length=3636):
    """
    Split the data into two separate dataset for instance into train and validation data.

    Parameters
    ----------
    dataset: tf.data.Dataset
      Dataset that should be splitted
    split_factor: float
      Percentage of the split (default: 0.3)
    dataset_length: int
      Length of the data that should be splitted (default: 3636)

    Returns
    -------
    returns: tf.data.Dataset
      Dataset with one hot encoded class labels
    """

    skip = int(dataset_length * (1 - split_factor))
    return dataset.take(skip), dataset.skip(skip)


def apply_data_augmentation(dataset, random_state=None):
    """
    Data augment the images in the dataset. Apply flip, rotation, crop, zoom

    Parameters
    ----------
    dataset: tf.data.Dataset
      Dataset that should be data augmented
    random_state: int
      Set a random state to get the same random augmentation every time

    Returns
    -------
    returns: tf.data.Dataset
      Dataset with data augmented data
    """

    pipe = tf.keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=random_state),
        keras.layers.experimental.preprocessing.RandomRotation(1, seed=random_state),
        keras.layers.RandomCrop(224, 224, seed=random_state),
        keras.layers.RandomZoom((-0.15, 0), seed=random_state)
    ])
    dataset = dataset.map(lambda x, y: (pipe(x, training=True), y))

    return dataset


def shuffle(dataset, dataset_length=3636):
    """
    Shuffle the data

    Parameters
    ----------
    dataset: tf.data.Dataset
      Dataset that should be shuffled
    dataset_length: int
      Defines the buffer size. For a complete shuffle this value should be
      greater or equals to the size of the dataset (default: 3636)

    Returns
    -------
    returns: tf.data.Dataset
      Dataset that is shuffled
    """

    return dataset.shuffle(dataset_length, seed=42, reshuffle_each_iteration=False)


def resize(dataset, size):
    """
    Resize dataset

    Parameters
    ----------
    dataset: tf.data.Dataset
      Dataset that should be resized
    size: (height, width) -> (int, int)
      New Size of the dataset


    Returns
    -------
    returns: tf.data.Dataset
      Dataset that is resized
    """

    return dataset.map(lambda x, y: (tf.image.resize(x, size), y))


def normalize(dataset):
    """
    Normalize the data between zero and one. CAUTION: This function should only be applied when training
    an own model. When training a pretrained model use the corresponding preprocess function of the model.

    Parameters
    ----------
    dataset: tf.data.Dataset
      Dataset that should be normalized


    Returns
    -------
    returns: tf.data.Dataset
      Dataset that is normalized between 0 and 1
    """

    return dataset.map(lambda x, y: (x / 255.0, y))


def preprocess_patches(data, size=transfer_learning.get_input_size()):
    """
    Apply the same preprocessing ont the images patches that where extracted by the
    sliding window approach

    Parameters
    ----------
    data: tf.data.Dataset
      Dataset with patches that should be preprocessed
    size: int
      Input size of the pretrained network to which the images should be resized to

    Returns
    -------
    returns: tf.data.Dataset
      Dataset with preprocessed patches
    """

    preprocessor = transfer_learning.get_preprocessor()
    if preprocessor:
        data = data.map(lambda x: preprocessor(x))

    if size:
        data = data.map(lambda x: tf.image.resize(x, size))

    return data


@tf.function
def get_balance_factor(label):
    """
    Function that can be used to balance the dataset by duplicating the data.
    This functions used the get_balance_factor() function above where the number
    of repeatings can be adjusted.

    Parameters
    ----------
    label: int

    Returns
    -------
    returns: int
      The amount of repetitions a specific class needs in order to balance the relevant data subset
    """

    if label == 1:
        return tf.cast(15, tf.int64)  # 135
    elif label == 2:
        return tf.cast(7, tf.int64)  # 140
    elif label == 3:
        return tf.cast(3, tf.int64)  # 111
    elif label == 4:
        return tf.cast(1, tf.int64)  # 140    ## 526
    else:
        return tf.cast(1, tf.int64)


def balance(data):
    """
    Function that can be used to balance the dataset by duplicating the data.
    This functions used the get_balance_factor() function above where the number
    of repeatings can be adjusted.

    Parameters
    ----------
    data: tf.data.Dataset
      Dataset that should be balanced

    Returns
    -------
    returns: tf.data.Dataset
      Balanced dataset
    """

    return data.flat_map(
        lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(get_balance_factor(y))  # get_balance_factor(y)
    )


def preprocess_train(data):
    """
    Function to preprocess the data

    Parameters
    ----------
    data: tf.data.Dataset
      Dataset that should be preprocessed

    Returns
    -------
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    """

    # select hyper-parameters
    split_factor = 0.2

    # use a smaller split_factor for the background data to balance it with the relevant dataset for better validation
    split_factor_bg = 0.0794

    # take background dataset
    count_bg = 3110
    background = data.take(count_bg)

    # shuffling background data
    background = shuffle(background, dataset_length=count_bg)
    background_train, background_val = split(background, dataset_length=count_bg, split_factor=split_factor_bg)

    # applying data augmentation
    background_train = apply_data_augmentation(background_train)
    background_val = apply_data_augmentation(background_val)

    # select relevant data
    rel = data.skip(3110)
    count_rel = 206

    rel = shuffle(rel, dataset_length=count_rel)

    # split train and val set before oversampling to avoid incorrect validations
    rel_train, rel_val = split(rel, dataset_length=count_rel, split_factor=split_factor)

    # oversample relevant data
    rel_repeat = 6
    rel_train = rel_train.repeat(rel_repeat)
    rel_val = rel_val.repeat(rel_repeat)
    count_rel *= rel_repeat

    # shuffling relevant data
    rel_train = shuffle(rel_train, dataset_length=int(count_rel * (1 - split_factor)))  # 1578
    rel_val = shuffle(rel_val, dataset_length=int(count_rel * split_factor))  # 1578

    # applying data augmentation
    rel_train = apply_data_augmentation(rel_train)
    rel_val = apply_data_augmentation(rel_val)

    # concatenating background and relevant data
    data_train = background_train.concatenate(rel_train)
    data_val = background_val.concatenate(rel_val)

    # final shuffling
    length = int(count_rel * (1 - split_factor)) + int(count_bg * (1 - split_factor_bg))
    data_train = shuffle(data_train, dataset_length=length)

    # apply preprocessor if needed
    preprocessor = transfer_learning.get_preprocessor()
    if preprocessor:
        data_train = data_train.map(lambda x, y: (preprocessor(x), y))
        data_val = data_val.map(lambda x, y: (preprocessor(x), y))

    # resize dataset if needed
    resizing = transfer_learning.get_input_size()
    if resizing:
        data_train = resize(data_train, resizing)
        data_val = resize(data_val, resizing)

    # apply one hot encoding
    data_train = one_hot_encode(data_train)
    data_val = one_hot_encode(data_val)

    # apply prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    data_train = data_train.prefetch(buffer_size=AUTOTUNE)
    data_val = data_val.prefetch(buffer_size=AUTOTUNE)

    # apply batch size
    data_train = data_train.batch(32)
    data_val = data_val.batch(32)

    return data_train, data_val
