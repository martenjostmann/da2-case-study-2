from skimage.util import view_as_windows
import numpy as np
import math
import tensorflow as tf


from drive.MyDrive.da2.src.d02_processing import preprocess_data

def find_classes(image, model, size, stride, shape=(256,256)):
  output = predict(image, model, size, stride, shape)
  output = np.stack(output, axis=1)

  pred = output[0]
  confidence = output[1]

  pred = pred.astype(int)

  #apply_nms(pred, confidence)

  return pred, confidence

def apply_nms(pred, confidence):
  for c in range(1,5):
    index = np.where(pred == c)
    boxes = []
    scores = []
    for i,j in zip(index[0], index[1]):

      score = confidence[j][i]
      scores.append(score)

      x1 = i*121
      x2 = x1+256
      y1 = j*121
      y2 = j+256
      box = [y1, x1, y2, x2]
      boxes.append(box)

    tf.image.non_max_suppression(boxes, scores, max_output_size=10).numpy()

def predict(image, model, size, stride, shape):
    '''
        Slide over the image and predict each pixels class
        
        ...
        
        Attributes
        ----------
        image : Numpy Array
            Image data that should be predicted
        model : Keras Model
            Model to predict classes
        padding : Int
            Zero Padding at the edges
        size: int
            Size of patches
        stide: int
            Step size
            
        Output
        ------
        ouput : List
            Segmented Image       
    '''
    
    # Initialize Output List (List to improve performance)
    output = []
    
    # Define image size
    image_size = image.shape[1]
    # Loop over the lines and exract a full line of patches
    for index, y in enumerate(range(0, image_size-255, stride)):
        
        num_patches = math.floor((image_size-size)/stride+1)

        # Print Progress
        if((index+1)%10==0):
            print("Iteration: {} von {}".format(index+1,num_patches))
        # Extract patches
        patches = view_as_windows(image[:,y:y+size,:], (size, size, 3), step=stride)
        
        # Reshape patches
        patches_reshape = np.reshape(patches, (num_patches, size, size, 3))

        dataset = tf.data.Dataset.from_tensors(patches_reshape)

        # preprocess patches
        dataset = preprocess_data.preprocess_patches(dataset, size=shape)
 
        # Append prediction to output list
        pred = model.predict(dataset)
        #pred = np.round(pred).astype(int).reshape(-1,)
        confidence = np.amax(pred, axis=1)
        pred = np.argmax(pred, axis=1)
        output.append([pred, confidence])
        
    return output
