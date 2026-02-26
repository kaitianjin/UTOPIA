#### Load package ####
import numpy as np
from .utils import load_image

def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (shape_ori + patch_size - 1) // patch_size * patch_size
    x_padded = np.pad(
        x,
        ((0, shape_ext[0] - x.shape[0]),
         (0, shape_ext[1] - x.shape[1]),
         (0, 0)),
        mode='edge')
    tiles_shape = shape_ext // patch_size
    patch_strides = (
        x_padded.strides[0] * patch_size,
        x_padded.strides[1] * patch_size,
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2])
    patches = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=(tiles_shape[0], tiles_shape[1], patch_size, patch_size, x.shape[2]),
        strides=patch_strides)
    patches = patches.reshape(-1, patch_size, patch_size, x.shape[2])
    shapes = dict(original=shape_ori, padded=shape_ext, tiles=tiles_shape)
    return patches, shapes


def compute_metrics(he, patch_size=16):
    he_tiles, shapes = patchify(he, patch_size=patch_size)

    he_std_image = np.std(he_tiles, axis=(1, 2, 3))
    he_std_image = he_std_image.reshape(shapes['tiles'])

    mean_rgb_per_patch = np.mean(he_tiles, axis=(1, 2))
    V_r = np.var(mean_rgb_per_patch[:, 0])
    V_g = np.var(mean_rgb_per_patch[:, 1])
    V_b = np.var(mean_rgb_per_patch[:, 2])
    numerators = (
        mean_rgb_per_patch[:, 0] * V_r +
        mean_rgb_per_patch[:, 1] * V_g +
        mean_rgb_per_patch[:, 2] * V_b)
    denominator = V_r + V_g + V_b
    z_v = numerators / denominator
    z_v_image = z_v.reshape(shapes['tiles'])

    # Normalize and compute ratio
    flattened_std = he_std_image.flatten()
    flattened_std = (flattened_std - flattened_std.min()) / (flattened_std.max() - flattened_std.min()) + 1
    he_std_norm_image = flattened_std.reshape(z_v_image.shape)

    flattened_mean = z_v_image.flatten()
    flattened_mean = (flattened_mean - flattened_mean.min()) / (flattened_mean.max() - flattened_mean.min()) + 1
    z_v_norm_image = flattened_mean.reshape(z_v_image.shape)

    ratio_norm = flattened_std / flattened_mean
    ratio_norm = (ratio_norm - ratio_norm.min()) / (ratio_norm.max() - ratio_norm.min())
    ratio_norm_image = ratio_norm.reshape(z_v_image.shape)

    return he_std_norm_image, he_std_image, z_v_norm_image, z_v_image, ratio_norm, ratio_norm_image







