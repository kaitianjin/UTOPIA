import argparse
import os
from time import time

from skimage.transform import rescale
import numpy as np

from ..utils.io import load_image, save_image


def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img

def get_image_filename(prefix):
    file_exists = False
    if_ome_tif = False 
    if_svs = False 
    for suffix in ['.jpg', '.png', '.tiff']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if os.path.exists(prefix + '.svs'): 
        filename = prefix + '.svs'
        file_exists = True 
        if_svs = True 
    if os.path.exists(prefix + '.ome.tif'):
        filename = prefix + '.ome.tif'
        file_exists = True 
        if_ome_tif = True 
    if os.path.exists(prefix + '.btf'):
        filename = prefix + '.btf'
        file_exists = True 
        if_ome_tif = True 
        
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename, if_ome_tif, if_svs 

def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img

def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--pixel_size_raw', type=float)
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    pixel_size = 0.5 # rescale to resolution of 0.5 um/pixel 
    scale = args.pixel_size_raw / pixel_size

    filename, if_ome_tif, if_svs = get_image_filename(
            os.path.join(args.data_path, 'HE', 'he-raw'))
    img = load_image(filename, if_ome_tif, if_svs)
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    t0 = time()
    img = rescale_image(img, scale)
    print(int(time() - t0), 'sec')
    img = img.astype(np.uint8)

    pad = 224
    img = adjust_margins(img, pad=pad, pad_value=255)
    save_image(img, os.path.join(args.data_path, 'HE', 'he.png'))

if __name__ == '__main__':
    main()
