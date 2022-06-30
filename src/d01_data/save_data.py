import csv

def save_bounding_boxes(predictions, padding):
  classes = ["pond", "pools", "solar", "trampoline"]

  header = ["class_label", "y_upper_left", "x_upper_left", "y_lower_right", "x_lower_right"]

  pred_output = []

  for y_i, y in enumerate(predictions):
    for x_i, pred in enumerate(y):
      if pred != 0:
        output = [classes[pred-1], y_i*padding, x_i*padding, y_i*padding+256, x_i*padding+256]
        pred_output.append(output)

  with open("predictions.csv", "w", newline='') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    writer.writerows(pred_output)
