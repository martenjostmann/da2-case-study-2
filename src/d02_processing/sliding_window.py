from skimage.util import view_as_windows
import numpy as np
import math
import tensorflow as tf

from src.d02_processing import preprocess


def find_classes(image, model, size, stride, shape=(256, 256)):
    """
    This function is used to find objects with the corresponding label and
    probability on big 8000x8000 image

    Parameters
    ----------
    image: nd.array
    image where objects should be found
    model: Keras model
    model that should be applied on the extracted patches
    size: int
    Size of patches
    stide: int
    Step size of the window
    shape:
    Size to which the patches should be reshaped to


    Returns
    -------
    pred: nd.array
    Predictions with the corresponding class label
    confidence: nd.array
    Probabilities of the prediction
    """

    output = predict(image, model, size, stride, shape)
    output = np.stack(output, axis=1)

    pred = output[0]
    confidence = output[1]

    pred = pred.astype(int)

    return pred, confidence


def apply_nms(pred, confidence, num_boxes=160):
    """
    Apply Non-maximum suppression (NMS) to reduce bounding boxes. In the end an own approach
    was applied.

    Parameters
    ----------
    pred: nd.array
      Predictions with the corresponding class label
    confidence: nd.array
      Probabilities of the prediction
    num_boxes: int
      Maximum number of boxes to be selected by Non-maximum suppression


    Returns
    -------
    final_boxes: nd.array
      Final Boxes with label and exact coordinates
    """

    final_boxes = []

    for c in range(1, 5):
        index = np.where(pred == c)
        boxes = []
        scores = []
        for i, j in zip(index[0], index[1]):
            score = confidence[j][i]
            scores.append(score)

            x1 = i * 121
            x2 = x1 + 256
            y1 = j * 121
            y2 = j + 256
            box = [y1, x1, y2, x2]
            boxes.append(box)

        merged = tf.image.non_max_suppression(boxes, scores, max_output_size=num_boxes).numpy()
        merged = tf.gather(boxes, merged)
        for box in merged:
            final_boxes.append((c, box[0] / 121, box[1] / 121))
    return final_boxes


def predict(image, model, size, stride, shape):
    """
    Slide over the image and predict each pixels class

    ...

    Attributes
    ----------
    image : Numpy Array
        Image data that should be predicted
    model : Keras Model
        Model to predict classes
    size: int
        Size of patches
    stide: int
        Step size of the window
    shape:
        Size to which the patches should be reshaped to

    Returns
    ------
    output : List
        Segmented Image
    """

    # Initialize Output List (List to improve performance)
    output = []

    # Define image size
    image_size = image.shape[1]
    # Loop over the lines and extract a full line of patches
    for index, y in enumerate(range(0, image_size - size - 1, stride)):

        num_patches = math.floor((image_size - size) / stride + 1)

        # Print Progress
        if (index + 1) % 10 == 0:
            print("Iteration: {} of {}".format(index + 1, num_patches))
        # Extract patches
        patches = view_as_windows(image[y:y + size, :, :], (size, size, 3), step=stride)

        # Reshape patches
        patches_reshape = np.reshape(patches, (num_patches, size, size, 3))

        dataset = tf.data.Dataset.from_tensors(patches_reshape)

        # preprocess patches
        dataset = preprocess.preprocess_patches(dataset, size=shape)

        # Append prediction to output list
        o_pred = model.predict(dataset)

        # binary
        # confidence = np.amax(o_pred)
        # pred = np.round(o_pred).astype(int).reshape(-1,)

        # non-binary
        confidence = np.amax(o_pred, axis=1)
        pred = np.argmax(o_pred, axis=1)

        # output.append(o_pred)
        output.append([pred, confidence])
    return output
