import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw(image, pred, padding):
  colors = ['r', 'g', 'b', 'y']

  fig, ax = plt.subplots()
  ax.imshow(image)

  for y_i, y in enumerate(pred):
    for x_i, x in enumerate(y):
      if x!=0:
        corner = (y_i*padding, x_i*padding)
        rect = patches.Rectangle((corner[1], corner[0]), 256, 256, linewidth=1, edgecolor=colors[int(x)-1], facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
  plt.axis('off')
  ponds = patches.Patch(color=colors[0], label='ponds')
  pools = patches.Patch(color=colors[1], label='pools')
  solar = patches.Patch(color=colors[2], label='solar')
  trampoline = patches.Patch(color=colors[3], label='trampoline')
  ax.legend(handles=[ponds, pools, solar, trampoline], bbox_to_anchor=(1, 1),
            bbox_transform=fig.transFigure)
  plt.savefig('my_fig.png', dpi=500)
  plt.show()