import os
import argparse
from typing import List

import numpy as np
from PIL import Image
import sys
from .io import load_mask, find_max_embedding_index


def crop_roi_from_wsi(
    wsi: np.ndarray,
    roi_top_left: list,
    roi_width: int,
    roi_height: int
) -> np.ndarray:
    """
    Crop a region of interest from a whole slide image.

    Args:
        wsi: A 2D or 3D numpy array (height, width) or (height, width, channels).
        roi_top_left: A list of two integers [row, col] specifying the top-left corner.
        roi_width: A positive integer specifying the width of the ROI.
        roi_height: A positive integer specifying the height of the ROI.

    Returns:
        The cropped region as a 2D or 3D numpy array (same dimensionality as input).

    Raises:
        ValueError: If inputs are invalid or ROI is out of bounds.
    """
    if wsi.ndim not in (2, 3):
        raise ValueError(f"wsi must be a 2D or 3D array, got {wsi.ndim}D")

    if len(roi_top_left) != 2:
        raise ValueError(f"roi_top_left must have exactly 2 elements, got {len(roi_top_left)}")

    if roi_width <= 0:
        raise ValueError(f"roi_width must be positive, got {roi_width}")

    if roi_height <= 0:
        raise ValueError(f"roi_height must be positive, got {roi_height}")

    row_start = roi_top_left[0]
    col_start = roi_top_left[1]
    row_end = row_start + roi_height
    col_end = col_start + roi_width

    wsi_height, wsi_width = wsi.shape[:2]

    if row_start < 0 or col_start < 0:
        raise ValueError(
            f"roi_top_left must be non-negative, got [{row_start}, {col_start}]"
        )

    if row_end > wsi_height or col_end > wsi_width:
        raise ValueError(
            f"ROI extends beyond wsi boundaries. "
            f"ROI ends at [{row_end}, {col_end}], but wsi shape is [{wsi_height}, {wsi_width}]"
        )

    if wsi.ndim == 2:
        return wsi[row_start:row_end, col_start:col_end]
    else:
        return wsi[row_start:row_end, col_start:col_end, :]


def crop_multiple_roi_from_wsi_embed(
    wsi_embed: np.ndarray,
    roi_top_left_list: List[List[int]],
    roi_width_list: List[int],
    roi_height_list: List[int],
    save_path: str
) -> None:
    """
    Crop multiple ROIs from a WSI embedding and save each to a file.

    Args:
        wsi_embed: A 3D numpy array (height, width, channels).
        roi_top_left_list: A list of [row, col] pairs specifying the top-left corner of each ROI.
        roi_width_list: A list of integers specifying the width of each ROI.
        roi_height_list: A list of integers specifying the height of each ROI.
        save_path: Directory path where the cropped ROI embeddings will be saved.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    num_rois = len(roi_top_left_list)

    if len(roi_width_list) != num_rois or len(roi_height_list) != num_rois:
        raise ValueError(
            f"Input lists must have the same length. Got {num_rois} top_left, "
            f"{len(roi_width_list)} widths, {len(roi_height_list)} heights."
        )

    os.makedirs(save_path, exist_ok=True)

    for roi_idx in range(num_rois):
        roi_embed = crop_roi_from_wsi(
            wsi_embed,
            roi_top_left_list[roi_idx],
            roi_width_list[roi_idx],
            roi_height_list[roi_idx]
        )
        output_file = os.path.join(save_path, f"embeddings_image_roi_{roi_idx}_scale_1.npy")
        np.save(output_file, roi_embed)

def crop_multiple_roi_from_wsi_mask(
    wsi_mask: np.ndarray,
    roi_top_left_list: List[List[int]],
    roi_width_list: List[int],
    roi_height_list: List[int],
    save_path: str
) -> None:
    """
    Crop multiple ROIs from a WSI mask and save each as a PNG file.

    Args:
        wsi_mask: A 2D numpy boolean array (height, width).
        roi_top_left_list: A list of [row, col] pairs specifying the top-left corner of each ROI.
        roi_width_list: A list of integers specifying the width of each ROI.
        roi_height_list: A list of integers specifying the height of each ROI.
        save_path: Directory path where the cropped ROI masks will be saved.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    num_rois = len(roi_top_left_list)

    if len(roi_width_list) != num_rois or len(roi_height_list) != num_rois:
        raise ValueError(
            f"Input lists must have the same length. Got {num_rois} top_left, "
            f"{len(roi_width_list)} widths, {len(roi_height_list)} heights."
        )

    os.makedirs(save_path, exist_ok=True)

    for roi_idx in range(num_rois):
        roi_mask = crop_roi_from_wsi(
            wsi_mask,
            roi_top_left_list[roi_idx],
            roi_width_list[roi_idx],
            roi_height_list[roi_idx]
        )
        # Convert boolean array to uint8 (0 or 255) for PNG saving
        roi_mask_uint8 = (roi_mask.astype(np.uint8)) * 255
        img = Image.fromarray(roi_mask_uint8, mode='L')
        output_file = os.path.join(save_path, f"mask-small_roi_{roi_idx}_scale_1.png")
        img.save(output_file)

def crop_multiple_roi_from_wsi_ct_mask(
    wsi_mask: np.ndarray,
    roi_top_left_list: List[List[int]],
    roi_width_list: List[int],
    roi_height_list: List[int],
    save_path: str
) -> None:
    """
    Crop multiple ROIs from a WSI mask and save each as a PNG file.

    Args:
        wsi_mask: A 2D numpy boolean array (height, width).
        roi_top_left_list: A list of [row, col] pairs specifying the top-left corner of each ROI.
        roi_width_list: A list of integers specifying the width of each ROI.
        roi_height_list: A list of integers specifying the height of each ROI.
        save_path: Directory path where the cropped ROI masks will be saved.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    num_rois = len(roi_top_left_list)

    if len(roi_width_list) != num_rois or len(roi_height_list) != num_rois:
        raise ValueError(
            f"Input lists must have the same length. Got {num_rois} top_left, "
            f"{len(roi_width_list)} widths, {len(roi_height_list)} heights."
        )

    os.makedirs(save_path, exist_ok=True)

    for roi_idx in range(num_rois):
        roi_mask = crop_roi_from_wsi(
            wsi_mask,
            roi_top_left_list[roi_idx],
            roi_width_list[roi_idx],
            roi_height_list[roi_idx]
        )
        # Convert boolean array to uint8 (0 or 255) for PNG saving
        roi_mask_uint8 = (roi_mask.astype(np.uint8)) * 255
        img = Image.fromarray(roi_mask_uint8, mode='L')
        output_file = os.path.join(save_path, f"ct-mask-small_roi_{roi_idx}_scale_1.png")
        img.save(output_file)

def crop_multiple_roi_from_wsi_ct_image(
    ct_image: np.ndarray,
    roi_top_left_list: List[List[int]],
    roi_width_list: List[int],
    roi_height_list: List[int],
    save_path: str
) -> None:
    """
    Crop multiple ROIs from a WSI cell type image and save each as a .npy file.

    Args:
        ct_image: A 2D numpy array of dtype int8 (height, width) indicating cell types.
        roi_top_left_list: A list of [row, col] pairs specifying the top-left corner of each ROI.
        roi_width_list: A list of integers specifying the width of each ROI.
        roi_height_list: A list of integers specifying the height of each ROI.
        save_path: Directory path where the cropped ROI cell type images will be saved.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    num_rois = len(roi_top_left_list)

    if len(roi_width_list) != num_rois or len(roi_height_list) != num_rois:
        raise ValueError(
            f"Input lists must have the same length. Got {num_rois} top_left, "
            f"{len(roi_width_list)} widths, {len(roi_height_list)} heights."
        )

    os.makedirs(save_path, exist_ok=True)

    for roi_idx in range(num_rois):
        roi_ct_image = crop_roi_from_wsi(
            ct_image,
            roi_top_left_list[roi_idx],
            roi_width_list[roi_idx],
            roi_height_list[roi_idx]
        )
        output_file = os.path.join(save_path, f"ct_image_roi_{roi_idx}_scale_1.npy")
        np.save(output_file, roi_ct_image)

def crop_multiple_roi_from_wsi_tr_image(
    tr_image: np.ndarray,
    roi_top_left_list: List[List[int]],
    roi_width_list: List[int],
    roi_height_list: List[int],
    save_path: str
) -> None:
    num_rois = len(roi_top_left_list)

    if len(roi_width_list) != num_rois or len(roi_height_list) != num_rois:
        raise ValueError(
            f"Input lists must have the same length. Got {num_rois} top_left, "
            f"{len(roi_width_list)} widths, {len(roi_height_list)} heights."
        )

    os.makedirs(save_path, exist_ok=True)

    for roi_idx in range(num_rois):
        roi_tr_image = crop_roi_from_wsi(
            tr_image,
            roi_top_left_list[roi_idx],
            roi_width_list[roi_idx],
            roi_height_list[roi_idx]
        )
        output_file = os.path.join(save_path, f"tr_image_roi_{roi_idx}_scale_1.npy")
        np.save(output_file, roi_tr_image)


def load_wsi_embeddings(data_path: str) -> np.ndarray:
    """
    Load WSI embeddings from multiple part files and reshape into a 3D array.

    Args:
        data_path: Path to directory containing embeddings_part_i.npy files and mask.

    Returns:
        3D numpy array of shape (W, L, 2048) containing the embeddings.
    """
    # Find max embedding index
    max_i = find_max_embedding_index(os.path.join(data_path,"embeddings"))

    # Load mask to determine output shape (W, L)
    mask_path = os.path.join(data_path, "mask", "mask-small.png")
    mask = load_mask(mask_path)
    W, L = mask.shape

    # Preallocate output array
    embeddings_3d = np.zeros((W, L, 2048), dtype=np.float32)
    embeddings_2d_view = embeddings_3d.reshape(W * L, 2048)

    # Load each part and copy directly into preallocated array
    current_idx = 0
    for i in range(max_i + 1):
        embedding_file = os.path.join(data_path, "embeddings", f"embeddings_part_{i}.npy")
        embeddings = np.load(embedding_file)  # Shape: (N, 2048)
        n = embeddings.shape[0]
        embeddings_2d_view[current_idx:current_idx + n] = embeddings
        current_idx += n

    return embeddings_3d

def create_roi_mask(roi_top_left_list: List[List[int]],
                    roi_width_list: List[int],
                    roi_height_list: List[int],
                    wsi_shape: tuple,
                    save_path: str) -> np.ndarray:
    roi_mask = np.zeros(wsi_shape, dtype=bool)
    for roi_idx in range(len(roi_top_left_list)):
        row_start = roi_top_left_list[roi_idx][0]
        col_start = roi_top_left_list[roi_idx][1]
        row_end = row_start + roi_height_list[roi_idx]
        col_end = col_start + roi_width_list[roi_idx]
        roi_mask[row_start:row_end, col_start:col_end] = True
    
    np.save(os.path.join(save_path, "roi_mask_scale_1.npy"), roi_mask)
    Image.fromarray(roi_mask.astype(np.uint8) * 255).save(os.path.join(save_path, "roi_mask_small.png"))
    return roi_mask

def main():
    parser = argparse.ArgumentParser(description="Crop ROIs from WSI embeddings and/or masks.")

    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to directory containing embeddings and mask files'
    )
    parser.add_argument(
        '--emb',
        action='store_true',
        default=False,
        help='Crop ROIs from WSI embeddings'
    )
    parser.add_argument(
        '--he_mask',
        action='store_true',
        default=False,
        help='Crop ROIs from WSI he mask'
    )
    parser.add_argument(
        '--ct_mask',
        action='store_true',
        default=False,
        help='Crop ROIs from WSI cell type mask'
    )
    parser.add_argument(
        '--ct_image',
        action='store_true',
        default=False,
        help='Crop ROIs from WSI cell type image (ct_image_scale_1.npy)'
    )
    parser.add_argument(
        '--tr_image',
        action='store_true',
        default=False,
        help='Crop ROIs from WSI transcripts image (tr_image_scale_1.npy)'
    )
    parser.add_argument(
        '--roi-top-left',
        type=str,
        required=True,
        help='ROI top-left coordinates as semicolon-separated pairs, e.g., "0,0;100,200"'
    )
    parser.add_argument(
        '--roi-width',
        type=str,
        required=True,
        help='ROI widths as comma-separated integers, e.g., "50,60"'
    )
    parser.add_argument(
        '--roi-height',
        type=str,
        required=True,
        help='ROI heights as comma-separated integers, e.g., "50,60"'
    )

    args = parser.parse_args()

    # Parse ROI parameters
    roi_top_left_list = [
        [int(x) for x in pair.split(',')]
        for pair in args.roi_top_left.split(';')
    ]
    roi_width_list = [int(x) for x in args.roi_width.split(',')]
    roi_height_list = [int(x) for x in args.roi_height.split(',')]


    mask_path = os.path.join(args.data_path, "mask", "mask-small.png")
    wsi_mask = load_mask(mask_path)
    create_roi_mask(
        roi_top_left_list,
        roi_width_list,
        roi_height_list,
        wsi_mask.shape,
        save_path=os.path.join(args.data_path, "mask"))
    

    if args.emb:
        print("Loading WSI embeddings...")
        wsi_embed = load_wsi_embeddings(args.data_path)
        print(f"Loaded embeddings with shape: {wsi_embed.shape}")
        print("Cropping ROIs from embeddings...")
        crop_multiple_roi_from_wsi_embed(
            wsi_embed,
            roi_top_left_list,
            roi_width_list,
            roi_height_list,
            save_path=os.path.join(args.data_path,"embeddings")
        )
        print(f"Saved cropped embeddings to: {os.path.join(args.data_path,'embeddings')}")

    if args.he_mask:
        mask_path = os.path.join(args.data_path, "mask", "mask-small.png")
        print(f"Loading H&E mask from: {mask_path}")
        wsi_mask = load_mask(mask_path)
        print(f"Loaded mask with shape: {wsi_mask.shape}")
        print("Cropping ROIs from H&E mask...")
        crop_multiple_roi_from_wsi_mask(
            wsi_mask,
            roi_top_left_list,
            roi_width_list,
            roi_height_list,
            save_path=os.path.join(args.data_path, "mask")
        )
        print(f"Saved cropped masks to: {os.path.join(args.data_path, 'mask')}")


    if args.ct_image:
        ct_image_path = os.path.join(args.data_path, "cell type", "ct_image_scale_1.npy")
        print(f"Loading cell type image from: {ct_image_path}")
        ct_image = np.load(ct_image_path)
        print(f"Loaded cell type image with shape: {ct_image.shape}, dtype: {ct_image.dtype}")
        print("Cropping ROIs from cell type image...")
        crop_multiple_roi_from_wsi_ct_image(
            ct_image,
            roi_top_left_list,
            roi_width_list,
            roi_height_list,
            save_path=os.path.join(args.data_path,"cell type")
        )
        print(f"Saved cropped cell type images to: {os.path.join(args.data_path,'cell type')}")

        ct_image_mask = ct_image > -1
        mask_dir = os.path.join(args.data_path, "mask")
        os.makedirs(mask_dir, exist_ok=True)
        Image.fromarray(ct_image_mask.astype(np.uint8) * 255).save(os.path.join(mask_dir, "ct-mask-small.png"))
        print(f"Saved cell type mask to: {os.path.join(mask_dir, 'ct-mask-small.png')}")

    if args.ct_mask:
        mask_path = os.path.join(args.data_path, "mask", "ct-mask-small.png")
        print(f"Loading mask from: {mask_path}")
        wsi_mask = load_mask(mask_path)
        print(f"Loaded mask with shape: {wsi_mask.shape}")
        print("Cropping ROIs from mask...")
        crop_multiple_roi_from_wsi_ct_mask(
            wsi_mask,
            roi_top_left_list,
            roi_width_list,
            roi_height_list,
            save_path=os.path.join(args.data_path, "mask")
        )
        print(f"Saved cropped masks to: {os.path.join(args.data_path, 'mask')}")



    if args.tr_image:
        tr_image_path = os.path.join(args.data_path, "transcripts", "tr_image_scale_1.npy")
        print(f"Loading transcripts image from: {tr_image_path}")
        tr_image = np.load(tr_image_path)
        print(f"Loaded transcripts image with shape: {tr_image.shape}, dtype: {tr_image.dtype}")
        print("Cropping ROIs from transcripts image...")
        crop_multiple_roi_from_wsi_tr_image(
            tr_image,
            roi_top_left_list,
            roi_width_list,
            roi_height_list,
            save_path=os.path.join(args.data_path,"transcripts")
        )
        print(f"Saved cropped transcripts images to: {os.path.join(args.data_path,'transcripts')}")


if __name__ == "__main__":
    sys.exit(main())
