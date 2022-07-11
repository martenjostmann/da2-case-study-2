import os
import tensorflow as tf
import tensorflow_datasets as tfds


def load_train_data(data_path=None):
    """
    Load train data from directory

    Parameters
    ----------
    data_path: String
      Path to the train data (default: None)

    Returns
    -------
    dataset: tf.data.Dataset
    """

    # define default path if it is not defined
    if data_path is None:
        data_path = os.path.join(os.getcwd(), 'data', 'training_patches')

    # load data as dataset with labels
    builder = tfds.folder_dataset.ImageFolder(data_path, shape=(256, 256, 3))

    dataset = builder.as_dataset(as_supervised=True, batch_size=-1)
    dataset = dataset['train']

    # sort data to be able to handle background data differently than relevant data in the following steps
    sort_order = list(tf.argsort(dataset[1]).numpy())

    # apply sort
    dataset = (tf.stack([dataset[0][idx] for idx in sort_order]), tf.stack([dataset[1][idx] for idx in sort_order]))

    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    return dataset


def load_train_data_as_binary(data_path=None):
    """
    Load train data in the binary form. This means that all backgroud classes correspond to class 0
    and every relevant data correspond to class 1. This approach was tried to first train a binary classifier
    which can distinguish between relevant and background data. But this approach did not perform well.

    Parameters
    ----------
    data_path: String
      Path to the train data (default: None)

    Returns
    -------
    binary_dataset: tf.data.Dataset
    """
    dataset = load_train_data(data_path)

    # binarize data
    binary_dataset = dataset.map(lambda x, y: (x, 1 if y > 0 else 0))

    return binary_dataset
