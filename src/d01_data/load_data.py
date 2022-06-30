import os
import tensorflow as tf
import tensorflow_datasets as tfds

def load_train_data():
    # define path
    data_base_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    data_path = os.path.join(data_base_path, 'training_patches')
    data_path=os.path.join(os.getcwd(), 'drive', 'MyDrive','da2','data','training_patches')
    
    # load data as dataset with labels
    builder = tfds.folder_dataset.ImageFolder(data_path, shape=(256, 256, 3))
    
    dataset = builder.as_dataset(as_supervised=True, batch_size=-1)
    dataset = dataset['train']
    
    # sort data
    sort_order = list(tf.argsort(dataset[1]).numpy())
    
    # apply sort
    dataset = (tf.stack([dataset[0][idx] for idx in sort_order]), tf.stack([dataset[1][idx] for idx in sort_order]))
    
    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    
    AUTOTUNE = tf.data.AUTOTUNE

    return dataset.prefetch(buffer_size=AUTOTUNE)
    

def load_train_data_as_binary():
    
    dataset = load_train_data()
    
    # binarize data
    binary_dataset = dataset.map(lambda x,y : (x, 1 if y>0 else 0))
    
    return binary_dataset