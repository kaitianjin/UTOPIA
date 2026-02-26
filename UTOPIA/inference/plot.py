import numpy as np
import matplotlib.pyplot as plt

def convert_confidence_to_rgb(confidence_image,
                              reserve_index_image,
                              total_true_image=None,
                              roi_mask=None,
                              true_type='magnitude',
                              positive_color=None):

    W, L = confidence_image.shape
    rgb = np.ones((W, L, 3), dtype=np.uint8) * 255  # default white

    heat_cmap = plt.get_cmap('hot')
    reserved = reserve_index_image.astype(bool)
    conf_rgb = (heat_cmap(confidence_image)[..., :3] * 255).astype(np.uint8)

    if roi_mask is None or total_true_image is None:
        rgb[reserved] = conf_rgb[reserved]
        return rgb

    # heat colormap for confidence outside roi_mask where reserve_index_image is True
    rgb[reserved & (~roi_mask)] = conf_rgb[reserved & (~roi_mask)]

    # turbo colormap for true values inside roi where reserve_index_image is True
    roi_and_reserved = roi_mask & reserved

    if true_type == 'magnitude':
        turbo_cmap = plt.get_cmap('turbo')
        true_rgb = (turbo_cmap(total_true_image)[..., :3] * 255).astype(np.uint8)
        rgb[roi_and_reserved] = true_rgb[roi_and_reserved]

    elif true_type == 'cell type':
        if positive_color is None:
            raise ValueError("positive_color must be provided when true_type='cell type'")
        pos_color = np.array(positive_color, dtype=np.uint8)
        rgb[roi_and_reserved] = np.array([225, 225, 225], dtype=np.uint8)
        rgb[roi_and_reserved & (total_true_image > 0)] = pos_color

    return rgb


def convert_binary_to_rgb(binary_image,
                          reserve_index_image,
                          positive_color=[255, 0, 0]):
    binary_image = binary_image.astype(bool)
    reserve_index_image = reserve_index_image.astype(bool)
    W, L = binary_image.shape
    rgb = np.ones((W, L, 3), dtype=np.uint8) * 255  # default white
    rgb[reserve_index_image] = [225,225,225]  # light gray for reserved areas
    rgb[reserve_index_image & binary_image] = positive_color
    return rgb



def convert_magnitude_to_rgb(magnitude_image,
                             reserve_index_image,
                             total_true_image=None,
                             roi_mask=None,
                             true_type='magnitude',
                             positive_color=None):

    W, L = magnitude_image.shape
    rgb = np.ones((W, L, 3), dtype=np.uint8) * 255  # default white

    turbo_cmap = plt.get_cmap('turbo')
    reserved = reserve_index_image.astype(bool)
    mag_rgb = (turbo_cmap(magnitude_image)[..., :3] * 255).astype(np.uint8)

    if roi_mask is None or total_true_image is None:
        rgb[reserved] = mag_rgb[reserved]
        return rgb

    # turbo colormap for magnitude outside roi_mask where reserve_index_image is True
    rgb[reserved & (~roi_mask)] = mag_rgb[reserved & (~roi_mask)]

    # inside roi_mask where reserve_index_image is True
    roi_and_reserved = roi_mask & reserved

    if true_type == 'magnitude':
        true_rgb = (turbo_cmap(total_true_image)[..., :3] * 255).astype(np.uint8)
        rgb[roi_and_reserved] = true_rgb[roi_and_reserved]

    elif true_type == 'cell type':
        if positive_color is None:
            raise ValueError("positive_color must be provided when true_type='cell type'")
        pos_color = np.array(positive_color, dtype=np.uint8)
        rgb[roi_and_reserved] = np.array([225, 225, 225], dtype=np.uint8)
        rgb[roi_and_reserved & total_true_image.astype(bool)] = pos_color

    return rgb


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
    