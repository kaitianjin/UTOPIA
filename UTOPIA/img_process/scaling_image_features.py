import numpy as np
import os
import argparse

from pyparsing import List
from ..utils.io import load_mask, find_max_embedding_index

from ..utils.sliding_window import sliding_window_3d, sliding_window
from PIL import Image
import sys

def reshape_image_features(data_path: str) -> np.ndarray:
    """
    Reshape embeddings from 2D parts into a 3D numpy array.

    Args:
        data_path: Path containing embedding files (embeddings_part_i.npy)
        scale: Scale factor (currently unused, reserved for future use)

    Returns:
        3D numpy array of shape (W, L, 2048) where W and L come from the mask shape
    """
    # Find the maximum embedding index
    max_i = find_max_embedding_index(os.path.join(data_path, "embeddings"))
    print(f"Found {max_i + 1} embedding files (0 to {max_i})")

    # Load mask to determine W and L
    mask_path = os.path.join(data_path, "mask", "mask-small.png")
    mask = load_mask(mask_path)
    W, L = mask.shape
    print(f"Mask shape: {W} x {L}")

    # Create 3D numpy array
    embeddings_3d = np.zeros((*mask.shape, 2048), dtype=np.float32)

    # Fill the 3D array with embeddings from parts
    current_pixel_idx = 0
    total_pixels = W * L

    for i in range(max_i + 1):
        embedding_file = os.path.join(data_path, "embeddings", f"embeddings_part_{i}.npy")

        if not os.path.exists(embedding_file):
            print(f"Warning: {embedding_file} not found, skipping...")
            continue

        print(f"Loading {embedding_file}...")
        embeddings_part = np.load(embedding_file)  # Shape: (N, 2048)
        n = embeddings_part.shape[0]

        # Calculate how many pixels we can fill from this part
        pixels_to_fill = min(n, total_pixels - current_pixel_idx)

        # Convert linear index range to 2D array indices
        for j in range(pixels_to_fill):
            linear_idx = current_pixel_idx + j
            row = linear_idx // L
            col = linear_idx % L
            embeddings_3d[row, col, :] = embeddings_part[j, :]

        current_pixel_idx += pixels_to_fill

        if current_pixel_idx >= total_pixels:
            break

    print(f"Filled {current_pixel_idx} pixels out of {total_pixels} total pixels")
    print(f"Final 3D array shape: {embeddings_3d.shape}")

    return embeddings_3d

def rescale_image_features(embed_image: np.ndarray, scale: int) -> np.ndarray:
    """
    Rescale image features using a sliding window operation.

    Args:
        embed_image: 3D numpy array of image embeddings
        scale: Integer scale factor for the window size

    Returns:
        Rescaled 3D numpy array
    """
    return sliding_window_3d(arr=embed_image, window_shape=(scale, scale), stride=(scale, scale), method="mean")

def find_roi_indices(data_path: str, scale: int) -> List[int]:
    """
    Find all ROI indices i in embeddings_image_roi_i_scale_j.npy files for a specific scale.

    Args:
        data_path: Path containing ROI embedding files
        scale: Scale factor

    Returns:
        List of unique ROI indices sorted in ascending order
    """
    roi_indices = set()
    for filename in os.listdir(data_path):
        if filename.startswith(f"embeddings_image_roi_") and filename.endswith(f"_scale_{scale}.npy"):
            # Extract the ROI index from filename
            # Format: embeddings_image_roi_i_scale_j.npy
            try:
                parts = filename.replace("embeddings_image_roi_", "").replace(".npy", "").split("_scale_")
                roi_idx = int(parts[0])
                roi_indices.add(roi_idx)
            except (ValueError, IndexError):
                continue

    if not roi_indices:
        raise FileNotFoundError(f"No ROI embedding files found for scale {scale} in {data_path}")

    return sorted(list(roi_indices))

def rescale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """
    Rescale a mask using a sliding window operation.

    Args:
        mask: 2D numpy array of the mask
        scale: Integer scale factor for the window size

    Returns:
        Rescaled 2D numpy array
    """
    return sliding_window(arr=mask, window_shape=(scale, scale), stride=(scale, scale), method="sum")

def process_image_scales(data_path: str, 
                         scales: list, 
                         emb: bool = False,
                         roi: bool = False, 
                         num_roi: int = 0,
                         wsi: bool = False, 
                         he_mask: bool = False, 
                         ct_mask: bool = False):
    
    """
    Process and rescale image embeddings at different scales.

    Args:
        data_path: Path containing embedding files
        scales: List of scale factors for rescaling
        roi: Whether to also process ROI embeddings
        wsi: WSI-specific processing flag
        mask: Whether to also process and rescale masks
    """
    if emb and wsi:
        # Create the 3D embed at scale 1
        print(f"Creating 3D embeddings at scale 1 from {os.path.join(data_path,'embeddings')}")
        embed_scale_1 = reshape_image_features(data_path)

        # Process each scale
        for scale in scales:
            print(f"\nProcessing scale {scale}")

            embed_rescaled = rescale_image_features(embed_scale_1, scale)
            output_path = os.path.join(data_path, "embeddings", f"embeddings_image_scale_{scale}.npy")
            print(f"Saving rescaled embeddings to {output_path}")
            np.save(output_path, embed_rescaled)

    #Process ROI embeddings if roi flag is set
    if emb and roi:
        print("\nProcessing ROI embeddings...")

        for roi_idx in range(num_roi):
            # Load ROI embedding at scale 1
            roi_scale_1_path = os.path.join(data_path, "embeddings", f"embeddings_image_roi_{roi_idx}_scale_1.npy")
            print(f"\nLoading ROI {roi_idx} from {roi_scale_1_path}")
            roi_embed_scale_1 = np.load(roi_scale_1_path)

            # Rescale for each scale
            for scale in scales:
                roi_embed_rescaled = rescale_image_features(roi_embed_scale_1, scale)
                roi_output_path = os.path.join(data_path, "embeddings", f"embeddings_image_roi_{roi_idx}_scale_{scale}.npy")
                print(f"Saving rescaled ROI {roi_idx} at scale {scale} to {roi_output_path}")
                np.save(roi_output_path, roi_embed_rescaled)

    # Process mask if mask flag is set
    if he_mask and wsi:
        print("\nProcessing mask...")
        mask_path = os.path.join(data_path, "mask", "mask-small.png")
        print(f"Loading mask from {mask_path}")
        mask_array = load_mask(mask_path)

        # Rescale for each scale
        for scale in scales:
            print(f"\nRescaling mask at scale {scale}")
            mask_rescaled = rescale_mask(mask_array, scale)
            mask_output_path = os.path.join(data_path, "mask", f"mask-small_scale_{scale}.png")
            print(f"Saving rescaled mask to {mask_output_path}")
            # Convert to image and save
            mask_img = Image.fromarray((mask_rescaled > 0).astype(np.uint8) * 255)
            mask_img.save(mask_output_path)

    # Process mask if mask flag is set
    if ct_mask and wsi:
        print("\nProcessing mask...")
        mask_path = os.path.join(data_path, "mask", "ct-mask-small.png")
        print(f"Loading mask from {mask_path}")
        mask_array = load_mask(mask_path)

        # Rescale for each scale
        for scale in scales:
            print(f"\nRescaling mask at scale {scale}")
            mask_rescaled = rescale_mask(mask_array, scale)
            mask_output_path = os.path.join(data_path, "mask", f"ct-mask-small_scale_{scale}.png")
            print(f"Saving rescaled mask to {mask_output_path}")
            # Convert to image and save
            mask_img = Image.fromarray((mask_rescaled > 0).astype(np.uint8) * 255)
            mask_img.save(mask_output_path)

    # Process ROI masks if both roi and mask flags are set
    if he_mask and roi:
        print("\nProcessing ROI masks...")

        for roi_idx in range(num_roi):
            # Load ROI mask at scale 1
            roi_mask_scale_1_path = os.path.join(data_path, "mask", f"mask-small_roi_{roi_idx}_scale_1.png")
            print(f"\nLoading ROI mask {roi_idx} from {roi_mask_scale_1_path}")
            roi_mask_array = load_mask(roi_mask_scale_1_path)

            # Rescale for each scale
            for scale in scales:
                print(f"Rescaling ROI mask {roi_idx} at scale {scale}")
                roi_mask_rescaled = rescale_mask(roi_mask_array, scale)
                roi_mask_output_path = os.path.join(data_path, "mask", f"mask-small_roi_{roi_idx}_scale_{scale}.png")
                print(f"Saving rescaled ROI mask to {roi_mask_output_path}")
                # Convert to image and save
                roi_mask_img = Image.fromarray((roi_mask_rescaled > 0).astype(np.uint8) * 255)
                roi_mask_img.save(roi_mask_output_path)

    # Process ROI masks if both roi and mask flags are set
    if ct_mask and roi:
        print("\nProcessing ROI masks...")

        for roi_idx in range(num_roi):
            # Load ROI mask at scale 1
            roi_mask_scale_1_path = os.path.join(data_path, "mask", f"ct-mask-small_roi_{roi_idx}_scale_1.png")
            print(f"\nLoading ROI mask {roi_idx} from {roi_mask_scale_1_path}")
            roi_mask_array = load_mask(roi_mask_scale_1_path)

            # Rescale for each scale
            for scale in scales:
                print(f"Rescaling ROI mask {roi_idx} at scale {scale}")
                roi_mask_rescaled = rescale_mask(roi_mask_array, scale)
                roi_mask_output_path = os.path.join(data_path, "mask", f"ct-mask-small_roi_{roi_idx}_scale_{scale}.png")
                print(f"Saving rescaled ROI mask to {roi_mask_output_path}")

                # Convert to image and save
                roi_mask_img = Image.fromarray((roi_mask_rescaled > 0).astype(np.uint8) * 255)
                roi_mask_img.save(roi_mask_output_path)

def main():
    """
    Command line interface for rescaling image features.
    """
    parser = argparse.ArgumentParser(description="Rescale image embeddings at different scales")
    parser.add_argument("--data_path", type=str, help="Path containing embedding files")
    parser.add_argument("--scale", type=int, nargs="+", default=[2],
                        help="Scale factor(s) for rescaling (default: 2)")
    parser.add_argument("--emb", action="store_true", default=False,
                        help="Process embeddings if flagged")
    parser.add_argument("--roi", action="store_true", default=False,
                        help="Process ROI if flagged")
    parser.add_argument("--num_roi", type=int, default=0,
                        help="Number of ROIs to process")
    parser.add_argument("--wsi", action="store_true", default=False,
                        help="Process WSI if flagged")
    parser.add_argument("--he_mask", action="store_true", default=False,
                        help="Process H&E mask if flagged")
    parser.add_argument("--ct_mask", action="store_true", default=False,
                        help="Process cell-type mask if flagged")
    args = parser.parse_args()

    process_image_scales(args.data_path, args.scale, args.emb, args.roi, args.num_roi, args.wsi, args.he_mask, args.ct_mask)

if __name__ == "__main__":
    sys.exit(main())
