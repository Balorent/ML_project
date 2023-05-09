import numpy as np
from scipy.ndimage import label
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import emnist
import format

img_name = 'images/four.jpeg'

# The original image
original_img = Image.open(img_name).convert('L')
# plot
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(7, 7)
ax1.imshow(original_img, cmap='gray')
ax1.set_title('Original picture')

# The binary image : same size as the original one
binary_img = format.binary_format(original_img)
binary_clustered_img, num_clusters = label(np.array(binary_img), structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
for i in range(num_clusters):
    cluster_i = (binary_clustered_img == i+1)
    if np.sum(cluster_i) > 500:
        fig_i, (ax3, ax4) = plt.subplots(1, 2)
        binary_img_i = Image.fromarray(cluster_i)

        # The standard image : 128x128 square image centered on the digit / letter
        standard_img_i = format.standard_format(binary_img_i)
        # plot
        ax3.imshow(standard_img_i.convert('L'), cmap='gray', vmin=0, vmax=255)
        ax3.set_title('Standardized picture')

        # The EMNIST image : 28x28 square image centered on the digit / letter
        emnist_img_i = format.emnist_format(standard_img_i)
        # plot
        ax4.imshow(emnist_img_i.convert('L'), cmap='gray', vmin=0, vmax=255)
        ax4.set_title('EMNIST picture')
    else:
        binary_clustered_img -= cluster_i*(i+1)
# plot
cmap = copy.copy(cm.get_cmap("gist_rainbow"))
cmap.set_bad(color='black')
ax2.imshow(np.ma.masked_where(binary_clustered_img == 0, binary_clustered_img), interpolation='nearest', cmap=cmap)
ax2.set_title('Binary clustered picture')

plt.show()
#nnrfio
#je suis las
#salut les gares
#test2