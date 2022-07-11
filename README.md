# Case Study: Satellite image object detection

The Goal of the Case Study 2 is to predict Objects like Ponds, Pools, Solar Panels and Trampolines within big Satellite images:

![Goal](goal.png)

## Requirements
---
A few additional modules are required. These modules are stored in `requirements.txt` and can be installed via the following command:  
`$ pip install -r requirements.txt`

In Google Colab this modules are already installed.

## File Structure

```
da2-case-study-2
│   README.md
│   requirements.txt
│
└───models
│   │
│   └───efficientnet_v2_best
│
└───notebooks
│   │   create_predictions.ipynb
|   |   training.ipynb
│
└───src
    │
    │
    └───d01_data
    │   │   load_data.py            <- Load train data
    |   |   save_data.py            <- Used to save the final bounding boxes with class label as csv file
    │
    └───d02_processing
    │   │   preprocess.py           <- Preprocessing of the train and val data as well as the preprocessing of the patches
    |   |   postprocess.py          <- Reduce the amount of bounding boxes
    |   |   sliding_window.py       <- Implemented the sliding window
    │
    └───d03_model
    │   │   transfer_learning.py    <- Get Preprocessing of pretrained models
    │
    └───d04_visualisation
    │   │   bounding_box.py         <- Visualize bounding boxes before and after reduction
    │
    └───d05_evalutation
    │   │   evaluate_predictions.py <- Calculate TP, FP, FN and F1-Score 
```

## Notebooks


In the notebooks directory the `create_predictions.ipynb` can be used to create predictions on every image that is placed inside the data folder. ALso the path of the directory can be adjusted inside of the notebook.

The `training.ipynb` was used to play around with the data and train different models and final evaluate them on the 8000x8000 images.
