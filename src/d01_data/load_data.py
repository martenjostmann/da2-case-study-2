import os
import tensorflow as tf
import tensorflow_datasets as tfds

def load_train_data():
    
    # define path
    data_base_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    data_path = os.path.join(data_base_path, 'training_patches')
    
    # load data as dataset with labels
    builder = tfds.folder_dataset.ImageFolder(data_path, shape=(256, 256, 3))
    
    dataset = builder.as_dataset(as_supervised=True)
    dataset = dataset['train']
    
    # shuffle data
    dataset = dataset.shuffle(len(dataset), seed=21,reshuffle_each_iteration=False)
    
    # one hot encode labels
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, 5)))
    
    # split data into train and val set
    train_dataset = dataset.take(int(len(dataset)*0.8))
    val_dataset = dataset.skip(int(len(dataset)*0.8))
    
    return train_dataset, val_dataset