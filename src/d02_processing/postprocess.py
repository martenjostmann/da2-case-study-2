import numpy as np


def get_grid(p, c, x, y, min_confidence=None, n_range_per_class=1):
  '''
    Search for potential patches with the same class to apply the vectorized 
    box reduction approach.
    
    ...
    
    Attributes
    ----------
    p : nd.array
        Predicted classes of the sliding window method
    c : nd.array
        Predicted confidence for each patch
    x : int
        The current x index where we should search for patches
    y : int
        The current y index where we should search for patches
    min_confidence : int
        The minimal confidence a box must have to be considered for the grid
    n_range_per_class : dict
        The range for each class to maybe consider more bounding boxes for a
        relevant object
        
    Returns
    ------
    returns : nd.array
        An array of potential patches including the current position
  '''
  patches = []
  clazz = p[y, x]
  n_range = n_range_per_class[clazz]

  for y_i in range(max(y - n_range, 0), min(y + n_range + 1, p.shape[0]), 1):
    for x_i in range(max(x - n_range, 0), min(x + n_range + 1, p.shape[1]), 1):
      if min_confidence is not None and c[y_i, x_i] < min_confidence:
        continue

      if p[y_i, x_i] == clazz:
        patches.append((x_i, y_i, c[y_i, x_i]))

  return np.array(patches)


def does_overlap(x, y, n_x, n_y, stride=121, width=256):
  '''
    Check whether two bounding boxes are overlapping or not.
    
    ...
    
    Attributes
    ----------
    x : int
        The x index of the first bounding box
    y : int
        The y index of the first bounding box
    n_x : int
        The x index of the second bounding box
    n_y : int
        The y index of the second bounding box
        
    Returns
    ------
    returns : bool
        True if it overlaps, false otherwise
  '''
  x_ = x * stride
  y_ = y * stride
  n_x_ = n_x * stride
  n_y_ = n_y * stride

  contains = lambda x1, x2: (x_ <= x1 and x1 <= x_ + width and 
                                y_ <= x2 and x2 <= y_ + width)

  return (contains(n_x_, n_y_) or 
          contains(n_x_ + width, n_y_) or 
          contains(n_x_, n_y_ + width) or 
          contains(n_x_ + width, n_y_ + width))


def calculate_position(grid):
  '''
    Core of the vectorized box reduction approach.
    
    ...
    
    Attributes
    ----------
    grid : int
        The x index of the first bounding box
    y : int
        The y index of the first bounding box
    n_x : int
        The x index of the second bounding box
    n_y : int
        The y index of the second bounding box
        
    Returns
    ------
    returns : bool
        True if it overlaps, false otherwise
  '''
  # It is not important to start with the patch in the middle from where we
  # searched for neighbors since our vector calculations do not have an order.
  
  # get first patch (c=confidence)
  x, y, c = grid[0]

  # check if we actually need to shift the current box
  N = len(grid) - 1
  if N == 0:
    return (x, y)

  # initialize shifted indices
  pos_x = x
  pos_y = y

  for n_x, n_y, n_c in grid[1:]:
    # start in the middle of both boxes
    mult = 0.5

    # consider confidences of both boxes and apply only half of the difference 
    # since we already are in the middle of the vector
    mult += (n_c - c) / 2

    # normalize the shift in consideration of the total amount of neighbors to
    # avoid leaving any predicted bounding boxes
    mult /= N

    # actual shifting into the direction of the box with the best confidence
    pos_x += (n_x - x) * mult
    pos_y += (n_y - y) * mult
  
  return (pos_x, pos_y)


def merge_predictions(p, c, min_confidence=None, n_range_per_class={1:2, 2:2, 3:1, 4:2}, stride=121, width=256):
  '''
    Core of the vectorized box reduction approach.
    
    ...
    
    Attributes
    ----------
    p : nd.array
        Predicted classes of the sliding window method
    c : nd.array
        Predicted confidence for each patch
    min_confidence : int
        The minimal confidence a box must have to be considered. Boxes with
        lower confidence will not be added to the output list.
    n_range_per_class : dict
        The range for each class to maybe consider more bounding boxes for a
        relevant object
    stride : int
        The stide which was used for the sliding window to calculate the exact
        positions of the final bounding boxes.
    width : int
        The width which was used for the sliding window to calculate the exact
        positions of the final bounding boxes.
        
    Returns
    ------
    bbs : list
        A list of shifted bounding boxes.
  '''
  p = np.copy(p)
  bbs=[]

  for y, l in enumerate(p):
    for x, value in enumerate(l):
      if min_confidence is not None and c[y, x] < min_confidence:
        continue

      clazz = p[y, x]
      if clazz != 0:
        grid = get_grid(p, c, x, y, min_confidence=min_confidence, n_range_per_class=n_range_per_class)
        if len(grid) > 0:
          pos_x, pos_y = calculate_position(grid)

          for n_x, n_y, _ in grid:
            p[int(n_y), int(n_x)] = 0

          bbs.append((clazz, pos_x, pos_y))

  return bbs
