import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd 
from utils import *


def plot_histology_clusters(he_clusters_image, 
                            num_he_clusters, 
                            ROI_top_left=None,
                            ROI_width=None, 
                            ROI_height=None,
                            color_list = None,
                            save_path=None):

    if color_list is None:
        color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                      [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                      [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                      [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                      [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                      [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                      [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
        
    
    he_clusters_image_rgb = 255*np.ones([np.shape(he_clusters_image)[0],np.shape(he_clusters_image)[1],3])
    for cluster in range(num_he_clusters):
        he_clusters_image_rgb[he_clusters_image==cluster] = color_list[cluster]
    he_clusters_image_rgb = np.array(he_clusters_image_rgb, dtype='int')
    
    plt.figure(figsize=(he_clusters_image.shape[1]//25, he_clusters_image.shape[0]//25))
    plt.title(f'Histology Clusters')
    plt.imshow(he_clusters_image_rgb)
    ax = plt.gca()

    if ROI_top_left is not None:
        rect = patches.Rectangle(
            xy=ROI_top_left,           # Top left corner
            width=ROI_width,           # Width
            height=ROI_height,         # Height
            linewidth=2,               # Line thickness
            edgecolor='red',           # Color of the rectangle
            facecolor='none'           # Transparent fill
        )
        ax.add_patch(rect)
    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    
    legend_elements = [patches.Patch(facecolor=np.array(color_list[i])/255, 
                                   label=f'Cluster {i}') 
                      for i in range(num_he_clusters)]
    plt.legend(handles=legend_elements, 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              borderaxespad=0.)
    if save_path is not None:
        plt.savefig(save_path, dpi=50)
    else:
        plt.show()





