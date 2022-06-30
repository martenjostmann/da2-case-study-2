from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf

def scheduler(epoch, lr):
    '''
        Simple lr_scheduler. Decreases the lr exponentially after 15 epochs
        
        Param:
            epoch: Current epoch
            lr: Current Learning Rate
        
        Return:
            Updatet Learning Rate
    '''
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def fit(model, train, val, class_weights, loss='categorical_crossentropy', optimizer=SGD, lr=0.001, lr_scheduler = True, early_stopping = True, patience=6, epochs=30):

    '''
      Compile and Train Model
      
      Param:
          model: Model that should be trained
          train: Training Data
          val: Validation Data
          loss: Loss function - Standard is categorical_crossentropy
          optimizer: Optimzer for the training - Standard is SGD (other option is Adam)
          lr: Learning Rate - Standard is 0.001
          lr_scheduler: Boolean if a Learning Rate Schedulder should be used - Standard is True
          early_stopping: Boolean if early stopping should be used - Standard is True
          patience: If early_stopping is true, than patience is the number of epochs that should be waited before stopping - Standard is 6
          epochs: Number of epochs
          class_weights: Weights for the classes
      
      Return:
          history: model_loss and model_accuracy for the train and val data
          model: Trained model
    '''
    # Compile Model
    model.compile(loss=loss, optimizer = optimizer(lr=lr), metrics=['accuracy'])
    
    if lr_scheduler:
        # Initialize Learning Rate Scheduler
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        if early_stopping:
            # Initialize Early Stopping
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            
            # Train Model with lr_scheduler und early stopping
            history = model.fit(train, epochs=epochs, validation_data=val, class_weight=class_weights, callbacks=[es_callback,lr_callback])
            
        else:
            # Train Model with lr_scheduler
            history = model.fit(train, epochs=epochs, validation_data=val, class_weight=class_weights, callbacks=[lr_callback])
    else:
        # Train Model
        history = model.fit(train, epochs=epochs, validation_data=val, class_weight=class_weights)

    
    return history, model