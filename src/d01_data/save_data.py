import csv


def save_boxes(boxes, path, stride=121, width=256):
    """
    Save bounding boxes into a csv file with the following structure (label, y_upper_left, x_upper_left, y_lower_right, x_lower_right)

    Parameters
    ----------
    boxes: List
    List of bounding boxes with the following structure (class_label, x_idx(float), y_idx(float)).
      In order to get the correct bounding box the indices has to be multiplied by the stride
    path: String
      Output including the name of the csv file
    stride: int
      Step size of the sliding window. (default: 121)
    width: int
      Width and also height of the window (default: 256)
    """

    out = []

    classes = ["pond", "pools", "solar", "trampoline"]
    header = ["label", "y_upper_left", "x_upper_left", "y_lower_right", "x_lower_right"]

    for p, x, y in boxes:
        out.append((
            classes[p - 1],
            int(y * stride),
            int(x * stride),
            int(y * stride + width),
            int(x * stride + width)
        ))

    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out)
