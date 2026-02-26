import pickle
import numpy as np
import os
from PIL import Image 

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, if_ome_tif=False, if_svs=False, verbose=True):
    if if_ome_tif:
        import tifffile 
        img = tifffile.imread(filename)
        img = np.array(img)
    elif if_svs: 
        import openslide 
        img = openslide.OpenSlide(filename)
        img = img.read_region((0, 0), 0, img.level_dimensions[0])
        img = img.convert('RGB')
        img = np.array(img)
    else:
        Image.MAX_IMAGE_PIXELS = None
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

def find_max_embedding_index(path: str) -> int:
    """
    Find the maximum index i in embeddings_part_i.npy files.

    Args:
        path: Path containing embedding files

    Returns:
        Maximum index i (N = max_i)
    """
    max_i = -1
    for filename in os.listdir(path):
        if filename.startswith("embeddings_part_") and filename.endswith(".npy"):
            # Extract the index from filename
            try:
                i = int(filename.replace("embeddings_part_", "").replace(".npy", ""))
                max_i = max(max_i, i)
            except ValueError:
                continue

    if max_i == -1:
        raise FileNotFoundError(f"No embedding files found in {path}")

    return max_i

def load_mask(mask_path: str) -> np.ndarray:
    """
    Load mask file as a 2D boolean numpy array.

    Args:
        mask_path: Path to mask-small.png file

    Returns:
        2D boolean numpy array of shape (W, L)
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)

    # Convert to boolean (assuming non-zero values are True)
    if mask_array.dtype == bool:
        return mask_array
    else:
        return mask_array > 0