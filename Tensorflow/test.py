import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nrow = 8
ncol = 8

fig = plt.figure(figsize=(ncol+1, nrow+1)) 

gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

for i in range(nrow):
    for j in range(ncol):
        im = np.ones((28,28), dtype=int)*255
        ax= plt.subplot(gs[i,j])
        ax.imshow(im, cmap='gray')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.axis('off')

plt.savefig(os.path.join(BASE_DIR, 'test.png'), bbox_inches = 'tight', pad_inches = 0)
