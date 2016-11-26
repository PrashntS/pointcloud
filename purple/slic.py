import matplotlib.pyplot as plt
import numpy as np
import skimage.io

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def main():
  img = skimage.io.imread('imsample/Turban_seller_at_Festival_au_Desert_near_Timbuktu,_Mali_2012_cropped.jpg')
  segments_slic = slic(img, n_segments=50, compactness=10, sigma=1)

  print("Slic number of segments: %d" % len(np.unique(segments_slic)))

  fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

  ax.imshow(mark_boundaries(img, segments_slic))
  ax.set_title("SLIC")

  plt.show()


if __name__ == '__main__':
  main()
