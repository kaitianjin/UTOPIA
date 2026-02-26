import itertools
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tifffile
import yaml
import torch
import random
import openslide

def set_seeds(seed):
    """
    Set seeds for all sources of randomness in Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use
    """
    
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Additional PyTorch settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

Image.MAX_IMAGE_PIXELS = None


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, if_ome_tif=False, if_svs=False, verbose=True):
    if if_ome_tif:
        img = tifffile.imread(filename)
        img = np.array(img)
    elif if_svs: 
        img = openslide.OpenSlide(filename)
        img = img.read_region((0, 0), 0, img.level_dimensions[0])
        img = img.convert('RGB')
        img = np.array(img)
    else:
        img = Image.open(filename)
        img = np.array(img)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]  # remove alpha channel
        if verbose:
            print(f'Image loaded from {filename}')
    return img


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename, quality=95)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]

def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    pad_w = shape_ext[0] - x.shape[0]
    pad_h = shape_ext[1] - x.shape[1]
    print(pad_w,pad_h)
    x = np.pad(x, ((0, pad_w),(0, pad_h),(0, 0)), mode='edge')
    patch_index_mask = np.zeros(np.shape(x)[:2])
    tiles_shape = np.array(x.shape[:2]) // patch_size
    tiles = []
    counter = 0
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size
        b0 = a0 + patch_size
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size
            b1 = a1 + patch_size
            tiles.append(x[a0:b0, a1:b1])
            patch_index_mask[a0:b0, a1:b1] = counter
            counter += 1

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    patch_index_mask = patch_index_mask[:np.shape(x)[0]-pad_w,:np.shape(x)[1]-pad_h]
    return tiles, shapes, patch_index_mask


def linear_line(x,x_vertex,m):
    return m*x - x_vertex 


# Define the linear boundary that starts at y = 0 and goes through (x_vertex, y_vertex) with slope m
# Input parameter m can be used to adjust the level of filtering

# Define the linear boundary function
def linear_bound(he_mean_image, he_std_image, m):

    mean_RGB = he_mean_image.copy().flatten()
    std_RGB = he_std_image.copy().flatten()

    coeffs = np.polyfit(mean_RGB, std_RGB, 2)  # Fit quadratic model based on input data
    a, b, c = coeffs
    
    # Calculate the x-coordinate of the vertex (peak)
    x_vertex = -b / (2 * a)
    
    # Calculate the y-intercept for the line with slope m that passes through (x_vertex, 0)
    y_intercept = -m * x_vertex
    
    print(f"Peak of the parabola occurs at Mean Intensity: {x_vertex:.2f}")
    
    return m*mean_RGB - x_vertex 

def visualize_linear_bound(he_mean_image, he_std_image, m):

    mean_RGB = he_mean_image.copy().flatten()
    std_RGB = he_std_image.copy().flatten()
    
    coeffs = np.polyfit(mean_RGB, std_RGB, 2)  # Fit a quadratic polynomial 
    a, b, c = coeffs  # Coefficients of the quadratic model
    # Step 2: Calculate the x-coordinate of the vertex (peak) of the parabola
    x_vertex = -b / (2 * a)
    # Step 3: Calculate the y-coordinate (standard deviation) at the vertex
    y_vertex = a * x_vertex**2 + b * x_vertex + c

    linear_boundary = linear_bound(he_mean_image, he_std_image, m)
    # Step 1: Create a mask to identify points below the linear boundary
    below_boundary_mask = std_RGB < linear_boundary
    # Step 2: Save the indices of the points below the linear boundary
    below_boundary_indices = np.where(below_boundary_mask)[0]

    # Step 3: Plot the points, color those below the boundary in red
    plt.scatter(mean_RGB[~below_boundary_mask], std_RGB[~below_boundary_mask], color='blue', s=.1)
    plt.scatter(mean_RGB[below_boundary_mask], std_RGB[below_boundary_mask], color='red', s=.1)

    # Plot the quadratic curve for visualization
    x_vals = np.linspace(np.min(mean_RGB), np.max(mean_RGB), 500)
    y_vals = a * x_vals**2 + b * x_vals + c
    plt.plot(x_vals, y_vals, 'g--', label='Fitted Quadratic')

    # Mark the vertex (peak)
    plt.scatter([x_vertex], [y_vertex], color='green', s=50, zorder=5)

    # Plot the linear boundary line
    x_line = np.linspace(x_vertex, np.max(mean_RGB), 500)
    y_line = linear_line(x_line, x_vertex, m)
    plt.plot(x_line, y_line, 'r--', label='Linear Boundary')
    plt.xlim(0, he_mean_image.max()+10)  
    plt.ylim(-5, he_std_image.max()+25)  

    # Plot settings
    plt.xlabel('Mean Intensity')
    plt.ylabel('Standard Deviation')
    plt.title(f'Linear Boundary with M = {m}')
    plt.legend()
    plt.show()