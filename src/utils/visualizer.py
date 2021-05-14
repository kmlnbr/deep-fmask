from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize,to_rgb
import matplotlib.patches as mpatches
import numpy as np
import os


# colors = ['black', 'saddlebrown', 'lightgray', 'gray','lavender','blue']
rgb=[]
RGB = [(0 ,0 ,0),(139, 69, 19),(245, 250, 178),(255, 0, 0 ),(85, 255, 255 ),(38 ,40, 210) ]
for i in RGB:
    rgb.append(tuple(j/255 for j in i))




# rgb = [to_rgb(i) for i in colors]

rgber = lambda t: rgb[t]
vfunc = np.vectorize(rgber)


color_labels = ['None', 'Rock', 'Cloud','Shadow','Ice','Water']
cmap = ListedColormap(rgb)

legend_elements = [mpatches.Patch(color=rgb[i], label=color_labels[i]) for i in range(6)]

def blend(class_mask):
    colr_fmask = vfunc(class_mask)
    colr_fmask = np.moveaxis(colr_fmask, 0, 2)

    return colr_fmask




