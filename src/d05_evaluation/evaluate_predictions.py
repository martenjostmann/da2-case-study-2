import csv, json, glob, cv2, random, os
from shapely.geometry import Polygon


def calc_performance(gt_path, pred_path, image_name=None, verbose=0):
    ground_truth = []
    predictions = []

    # Create default performance values
    performances = {
        'file': image_name,
        'tp': 0,
        'fn': 0,
        'fp': 0,
        'f1': 0,
    }

    ## Load ground truth
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: int(row[k]) if k != 'label' else row[k] for k in row.keys()}
            ground_truth.append(row)

    ## load predictions if path exists
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k: int(row[k]) if k != 'label' else row[k] for k in row.keys()}
                predictions.append(row)

    # Number of false positives equals number of left predictions
    performances['fp'] = max(len(predictions) - len(ground_truth), 0)

    for j, gt in enumerate(ground_truth):
        gt_box = Polygon([(gt['y_upper_left'], gt['x_upper_left']),
                          (gt['y_upper_left'], gt['x_lower_right']),
                          (gt['y_lower_right'], gt['x_lower_right']),
                          (gt['y_lower_right'], gt['x_upper_left'])])

        if gt_box.area != (256. * 256.):
            print(f'### Warning {j}: false ground truth shape of {gt_box.area} detected in {image_name}!')
            print(gt['y_lower_right'] - gt['y_upper_left'], gt['x_lower_right'] - gt['x_upper_left'])

        best_found_iou = (None, 0.)  # (idx, IoU)
        for i, pred in enumerate(predictions):
            if gt['label'] == pred['label']:
                pred_box = Polygon([(pred['y_upper_left'], pred['x_upper_left']),
                                    (pred['y_upper_left'], pred['x_lower_right']),
                                    (pred['y_lower_right'], pred['x_lower_right']),
                                    (pred['y_lower_right'], pred['x_upper_left'])])

                if pred_box.area != (256. * 256.):
                    print(f'### Warning {i}: false predicted shape of {pred_box.area} detected in {image_name}!')
                    print(pred['y_lower_right'] - pred['y_upper_left'], pred['x_lower_right'] - pred['x_upper_left'])

                ## Calculate IoU
                next_iou = (gt_box.intersection(pred_box).area + 1) / (gt_box.union(pred_box).area + 1)

                # If the next found IoU is larger than the previous found IoU -> override
                if next_iou > best_found_iou[1]:
                    best_found_iou = (i, next_iou)

        ## Append metric. If IoU is larger 0.5, then its a true positive, else false negative
        if best_found_iou[0] is not None and best_found_iou[1] >= 0.5:
            del predictions[best_found_iou[0]]  # Remove prediction from list!
            performances['tp'] += 1  # Increase number of True Positives
            if verbose == 1:
                print(f'Found correct prediction with IoU of {round(best_found_iou[1], 3)} and label {gt["label"]}!')
        else:
            performances['fn'] += 1  # Increase number of False Negatives
            if verbose == 1:
                print(f'Found false prediction with IoU of {round(best_found_iou[1], 3)} and label {gt["label"]}!')

    ## Calculate F1-Score
    performances['f1'] = (performances['tp'] + 1e-8) / \
                         (performances['tp'] + 0.5 * (performances['fp'] + performances['fn']) + 1e-8)
    return performances


if __name__ == "__main__":
    path = 'validation_data'  # Change if needed

    ## Iterate over all validation images
    for image_path in glob.glob(path + '/*.png'):
        image_name = image_path.split('/')[-1]
        gt_path = image_path[:-4] + '.csv'  # Ground Truth path
        pred_path = image_path[:-4] + '_prediction.csv'  # Prediction path
        performance = calc_performance(gt_path, pred_path, image_name)
        print(performance)
