import numpy as np

color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                      [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                      [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                      [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                      [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                      [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                      [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]

def convert_histology_to_rgb(he_clusters_image,
                             num_he_clusters):

    if num_he_clusters > len(color_list):
        raise ValueError(f"num_he_clusters exceeds the number of available colors in color_list ({len(color_list)}). \
                         Please modify `color_list` in UTOPIA.img_process.plot.py.")
    
    W, L = he_clusters_image.shape
    rgb = np.ones((W, L, 3), dtype=np.uint8) * 255  # default white

    for cluster_idx in range(num_he_clusters):
        color = color_list[cluster_idx]
        rgb[he_clusters_image == cluster_idx] = color

    return rgb.astype(np.uint8)


def plot_rgb_image(rgb_image, 
                   save_path=None, 
                   ROI_top_lefts=None,
                   ROI_widths=None, 
                   ROI_heights=None,
                   linewidth=2, 
                   edgecolor='red'):  

    from PIL import Image, ImageDraw
        
    img = Image.fromarray(rgb_image.astype(np.uint8))
        
    # Draw ROI rectangles if needed
    if ROI_top_lefts is not None:
        draw = ImageDraw.Draw(img)
        for i, ROI_top_left in enumerate(ROI_top_lefts):
            x, y = ROI_top_left
            x2 = x + ROI_widths[i]
            y2 = y + ROI_heights[i]
            draw.rectangle([x, y, x2, y2], outline=edgecolor, width=linewidth)
    
    img.save(save_path)
    return
    