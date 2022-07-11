import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw(image, pred, stride, save=False):
  """
  Plot the image with found bounding boxes before bounding box reduction

  Parameters
  ----------
  image: nd.array
    image where objects should be found
  pred: nd.array
    Predictions with the corresponding class label
  stride: int
    Step size of the window
  save: bool
    Should the resulting image be saved to a file (default False)

  """


  colors = ['y', 'r', 'b', 'g']

  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(image)

  for y_i, y in enumerate(pred):
    for x_i, x in enumerate(y):
      if x!=0:
        corner = (x_i*stride, y_i*stride)
        rect = patches.Rectangle(corner, 256, 256, linewidth=0.5, edgecolor=colors[int(x)-1], facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

  plt.axis('off')
  ponds = patches.Patch(color=colors[0], label='ponds')
  pools = patches.Patch(color=colors[1], label='pools')
  solar = patches.Patch(color=colors[2], label='solar')
  trampoline = patches.Patch(color=colors[3], label='trampoline')
  ax.legend(handles=[ponds, pools, solar, trampoline], bbox_to_anchor=(1, 1),
            bbox_transform=fig.transFigure)
  if save:
    plt.savefig('my_fig.png', dpi=500)
  plt.show()


def draw_boxes(image, boxes, stride, save=False):
  """
  Plot the image with found bounding boxes after bounding box reduction

  Parameters
  ----------
  image: nd.array
    image where objects should be found
  pred: nd.array
    Predictions with the corresponding class label
  stride: int
    Step size of the window
  save: bool
    Should the resulting image be saved to a file (default False)

  """

  colors = ['y', 'r', 'b', 'g']

  fig, ax = plt.subplots(figsize=(20, 20))
  ax.imshow(image)

  for i, (clazz, x, y) in enumerate(boxes):
    rect = patches.Rectangle((x * stride, y * stride), 256, 256, linewidth=0.5, edgecolor=colors[int(clazz)-1], facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

  plt.axis('off')
  ponds = patches.Patch(color=colors[0], label='ponds')
  pools = patches.Patch(color=colors[1], label='pools')
  solar = patches.Patch(color=colors[2], label='solar')
  trampoline = patches.Patch(color=colors[3], label='trampoline')
  ax.legend(handles=[ponds, pools, solar, trampoline], bbox_to_anchor=(1, 1),
            bbox_transform=fig.transFigure)

  if save:
    plt.savefig('my_fig_nms.png', dpi=500)
  plt.show()