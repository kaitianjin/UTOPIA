import numpy as np
import os
import argparse
from typing import List, Union, Tuple
from sklearn.decomposition import PCA
import sys
from ..utils.io import save_pickle, load_pickle, load_mask, find_max_embedding_index


def reduce_image_features_roi(
    data_paths: List[str],
    global_dim: int,
    local_dim: int,
    scale: int,
    delete_raw_features: bool = False
) -> None:
    """
    Process ROI embeddings using pre-fitted PCA encoders.

    Args:
        data_path: List of paths containing ROI embedding files
        global_dim: Target dimension for global features
        local_dim: Target dimension for local features
        scale: Scale factor for embeddings
        delete_raw_features: If True, delete raw embedding files after processing
    """

    total_dim = global_dim + local_dim

    for path in data_paths:
        print(f"\n{'=' * 80}")
        print(f"Processing ROI embeddings for: {path}")
        print(f"{'=' * 80}")

        embedding_path = os.path.join(path, "embeddings")

        # Load PCA encoders
        pca_file = os.path.join(embedding_path, f"pca_dim_{total_dim}_scale_{scale}.pkl")
        if not os.path.exists(pca_file):
            raise FileNotFoundError(f"PCA encoder file not found: {pca_file}")

        print(f"Loading PCA encoders from: {pca_file}")
        pca_data = load_pickle(pca_file, verbose=False)
        pca_global = pca_data['pca_global']
        pca_local = pca_data['pca_local']

        # Find all ROI indices for this scale
        roi_indices = find_roi_indices(embedding_path, scale)
        print(f"Found {len(roi_indices)} ROIs: {roi_indices}")

        # Process each ROI
        for roi_idx in roi_indices:
            print(f"\n{'-' * 80}")
            print(f"Processing ROI {roi_idx}")
            print(f"{'-' * 80}")

            # Load ROI embeddings
            embedding_file = os.path.join(embedding_path, f"embeddings_image_roi_{roi_idx}_scale_{scale}.npy")
            print(f"Loading: {embedding_file}")
            embeddings_3d = np.load(embedding_file)  # Shape: (W, L, 2048)
            W, L, feature_dim = embeddings_3d.shape
            print(f"Loaded 3D embeddings with shape: {embeddings_3d.shape}")

            if feature_dim != 2048:
                raise ValueError(f"Expected feature dimension 2048, got {feature_dim}")

            # Load mask
            mask_path = os.path.join(path, "mask", f"mask-small_roi_{roi_idx}_scale_{scale}.png")
            print(f"Loading mask: {mask_path}")
            mask = load_mask(mask_path)

            if mask.shape != (W, L):
                raise ValueError(f"Mask shape {mask.shape} does not match embeddings shape ({W}, {L})")

            num_masked_pixels = np.sum(mask)
            print(f"Mask shape: {W} x {L}")
            print(f"Number of masked pixels: {num_masked_pixels}")

            # Extract masked embeddings
            embeddings_2d = embeddings_3d.reshape(W * L, 2048)
            mask_flat = mask.flatten()
            masked_embeddings = embeddings_2d[mask_flat]  # Shape: (num_masked_pixels, 2048)

            # Split into global and local features
            global_features = masked_embeddings[:, 0:1024]
            local_features = masked_embeddings[:, 1024:2048]

            # Transform using pre-fitted PCA encoders
            print(f"Transforming global features: {global_features.shape[0]} samples")
            reduced_global = pca_global.transform(global_features)

            print(f"Transforming local features: {local_features.shape[0]} samples")
            reduced_local = pca_local.transform(local_features)

            # Concatenate reduced features
            reduced_features = np.hstack([reduced_global, reduced_local])
            print(f"Reduced features shape: {reduced_features.shape}")

            # Save reduced ROI embeddings
            output_file = os.path.join(embedding_path, f"embeddings_roi_{roi_idx}_scale_{scale}.npy")
            np.save(output_file, reduced_features)

            print(f"Saved to: {output_file}")
            print(f"  Shape: {reduced_features.shape}")
            print(f"  Size: {reduced_features.nbytes / (1024**2):.2f} MB")

            # Delete raw embedding file if requested
            if delete_raw_features:
                if os.path.exists(embedding_file):
                    os.remove(embedding_file)
                    print(f"Deleted raw embedding file: {embedding_file}")

    print(f"\n{'=' * 80}")
    print("DONE! All ROI features reduced and saved.")
    print(f"{'=' * 80}")


def apply_pca_reduction(
    features: np.ndarray,
    n_components: int,
    feature_name: str = "features"
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction to features.

    Args:
        features: Input features array of shape (n_samples, n_features)
        n_components: Target number of dimensions
        feature_name: Name of the feature type for logging

    Returns:
        Tuple of (reduced_features, pca_model)
    """
    print(f"\nApplying PCA to {feature_name} ({features.shape[1]} -> {n_components})...")

    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    explained_var = np.sum(pca.explained_variance_ratio_)

    print(f"Reduced {feature_name} shape: {reduced_features.shape}")
    print(f"Explained variance ratio: {explained_var:.4f}")

    return reduced_features, pca


def reduce_image_features_wsi(
    data_paths: Union[str, List[str]],
    global_dim: int = 100,
    local_dim: int = 100,
    scale: int = 1,
    delete_raw_features: bool = False
) -> None:
    """
    Reduce image feature dimensions from 2048 to target dimensions.

    This function:
    1. Extracts and reduces GLOBAL features first (1024 -> global_dim)
    2. Then extracts and reduces LOCAL features (1024 -> local_dim)
    3. Concatenates reduced global and local features
    4. Saves reduced features back to corresponding data_path

    Args:
        data_paths: Single path or list of paths containing embedding files
        global_dim: Target dimension for global features (default: 100)
        local_dim: Target dimension for local features (default: 100)
        scale: Scale factor for embeddings (default: 1)
        delete_raw_features: If True and scale != 1, delete raw embedding files after processing
    """
    # Ensure data_paths is a list
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    # Store metadata for each path
    path_data = []

    # ========================================================================
    # STEP 1: Extract and reduce GLOBAL features
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Processing GLOBAL features")
    print("=" * 80)

    all_global_features = []

    for path in data_paths:
        print(f"\nProcessing path: {path}")

        if scale == 1:
            embedding_path = os.path.join(path,"embeddings")
            # Original behavior: read from embeddings_part_i.npy files
            # Find the maximum i in embeddings_part_i.npy files
            max_i = find_max_embedding_index(embedding_path)
            print(f"Found {max_i + 1} embedding files (0 to {max_i})")

            # Load mask file
            mask_path = os.path.join(path, "mask", "mask-small.png")
            mask = load_mask(mask_path)
            W, L = mask.shape
            total_pixels = W * L
            num_masked_pixels = np.sum(mask)

            print(f"Mask shape: {W} x {L} = {total_pixels} total pixels")
            print(f"Number of masked pixels: {num_masked_pixels}")

            # Extract global features
            print("Extracting global features...")
            global_features = extract_masked_features(
                embedding_path, max_i, mask, feature_type='global'
            )
            print(f"Extracted global features shape: {global_features.shape}")

            # Store metadata
            path_data.append({
                'path': path,
                'mask': mask,
                'max_i': max_i,
                'num_pixels': num_masked_pixels,
                'scale': scale
            })

            # Accumulate global features
            all_global_features.append(global_features)
        else:
            # New behavior for scale != 1: read from embeddings_image_scale_j.npy
            embedding_path = os.path.join(path,"embeddings")
            embedding_file = os.path.join(embedding_path, f"embeddings_image_scale_{scale}.npy")
            print(f"Loading {embedding_file}...")

            embeddings_3d = np.load(embedding_file)  # Shape: (W, L, 2048)
            print(f"Loaded 3D embeddings with shape: {embeddings_3d.shape}")
            W, L, feature_dim = embeddings_3d.shape

            if feature_dim != 2048:
                raise ValueError(f"Expected feature dimension 2048, got {feature_dim}")

            mask_path = os.path.join(path, "mask", f"mask-small_scale_{scale}.png")
            mask = load_mask(mask_path)

            if mask.shape != (W, L):
                raise ValueError(f"Mask shape {mask.shape} does not match embeddings shape ({W}, {L})")

            num_masked_pixels = np.sum(mask)
            print(f"Mask shape: {W} x {L}")
            print(f"Number of masked pixels: {num_masked_pixels}")

            # Extract global features (first 1024 dimensions) for masked pixels
            print("Extracting global features...")
            embeddings_2d = embeddings_3d.reshape(W * L, 2048)
            mask_flat = mask.flatten()
            masked_embeddings = embeddings_2d[mask_flat]
            global_features = masked_embeddings[:, 0:1024]
            print(f"Extracted global features shape: {global_features.shape}")

            # Store metadata
            path_data.append({
                'path': path,
                'mask': mask,
                'embeddings_3d': embeddings_3d,
                'num_pixels': num_masked_pixels,
                'scale': scale,
                'embedding_file': embedding_file
            })

            # Accumulate global features
            all_global_features.append(global_features)

    # Concatenate all global features
    print("\n" + "=" * 80)
    print("Concatenating all global features")
    print("=" * 80)
    all_global_features = np.vstack(all_global_features)
    print(f"Total global features shape: {all_global_features.shape}")

    # Apply PCA to global features
    print("\n" + "=" * 80)
    print(f"STEP 2: Applying PCA to global features")
    print("=" * 80)
    reduced_global, pca_global = apply_pca_reduction(
        all_global_features, global_dim, "global features"
    )

    # Free memory - no longer need unreduced global features
    del all_global_features

    # ========================================================================
    # STEP 3: Extract and reduce LOCAL features
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Processing LOCAL features")
    print("=" * 80)

    all_local_features = []
    
    for data in path_data:
        path = data['path']
        mask = data['mask']
        scale = data['scale']
        embedding_path = os.path.join(path,"embeddings")

        print(f"\nProcessing path: {path}")
        print("Extracting local features...")
    
        if scale == 1:
            max_i = data['max_i']
            local_features = extract_masked_features(
                embedding_path, max_i, mask, feature_type='local'
            )
        else:
            # For scale != 1, extract local features from the 3D embeddings
            embeddings_3d = data['embeddings_3d']
            W, L, _ = embeddings_3d.shape
            embeddings_2d = embeddings_3d.reshape(W * L, 2048)
            mask_flat = mask.flatten()
            masked_embeddings = embeddings_2d[mask_flat]
            local_features = masked_embeddings[:, 1024:2048]

        print(f"Extracted local features shape: {local_features.shape}")

        # Accumulate local features
        all_local_features.append(local_features)

    # Concatenate all local features
    print("\n" + "=" * 80)
    print("Concatenating all local features")
    print("=" * 80)
    all_local_features = np.vstack(all_local_features)
    print(f"Total local features shape: {all_local_features.shape}")

    # Apply PCA to local features
    print("\n" + "=" * 80)
    print(f"STEP 4: Applying PCA to local features")
    print("=" * 80)
    reduced_local, pca_local = apply_pca_reduction(
        all_local_features, local_dim, "local features"
    )

    # Free memory - no longer need unreduced local features
    del all_local_features

    # ========================================================================
    # STEP 5: Concatenate reduced features
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"STEP 5: Concatenating reduced features ({global_dim} + {local_dim} = {global_dim + local_dim})")
    print("=" * 80)

    all_reduced_features = np.hstack([reduced_global, reduced_local])
    print(f"Final concatenated features shape: {all_reduced_features.shape}")

    # Free memory
    del reduced_global, reduced_local

    # ========================================================================
    # STEP 6: Save reduced features and PCA encoders back to each path
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Saving reduced features and PCA encoders to respective paths")
    print("=" * 80)

    current_idx = 0
    total_dim = global_dim + local_dim

    for data in path_data:
        path = data['path']
        num_pixels = data['num_pixels']
        scale = data['scale']

        embedding_path = os.path.join(path,"embeddings")

        # Extract features for this path
        path_features = all_reduced_features[current_idx:current_idx + num_pixels]
        current_idx += num_pixels

        # Save reduced features
        output_file = os.path.join(embedding_path, f"embeddings_scale_{scale}.npy")
        np.save(output_file, path_features)

        print(f"\nSaved reduced features to: {output_file}")
        print(f"  Shape: {path_features.shape}")
        print(f"  Size: {path_features.nbytes / (1024**2):.2f} MB")

        # Save PCA encoders
        pca_file = os.path.join(embedding_path, f"pca_dim_{total_dim}_scale_{scale}.pkl")
        pca_data = {
            'pca_global': pca_global,
            'pca_local': pca_local,
            'global_dim': global_dim,
            'local_dim': local_dim,
            'scale': scale
        }
        save_pickle(pca_data, pca_file)

        print(f"Saved PCA encoders to: {pca_file}")
        print(f"  Global PCA: 1024 -> {global_dim}")
        print(f"  Local PCA: 1024 -> {local_dim}")

        # Delete raw embedding file if requested and scale != 1
        if delete_raw_features and scale != 1:
            embedding_file = data.get('embedding_file')
            if embedding_file and os.path.exists(embedding_file):
                os.remove(embedding_file)
                print(f"Deleted raw embedding file: {embedding_file}")

    print("\n" + "=" * 80)
    print("DONE! All features reduced and saved.")
    print("=" * 80)


def reduce_image_features_wsi_wrapper(
    data_paths: Union[str, List[str]],
    global_dim: int = 100,
    local_dim: int = 100,
    scales: List[int] = [1],
    delete_raw_features: bool = False
) -> None:
    """
    Wrapper function to process WSI embeddings for multiple scales.

    Args:
        data_path: Single path or list of paths containing embedding files
        global_dim: Target dimension for global features (default: 100)
        local_dim: Target dimension for local features (default: 100)
        scales: List of scale factors for embeddings (default: [1])
        delete_raw_features: If True and scale != 1, delete raw embedding files after processing
    """
    if isinstance(scales, int):
        scales = [scales]
    for scale in scales:
        print(f"\n{'#' * 80}")
        print(f"Processing WSI embeddings for scale: {scale}")
        print(f"{'#' * 80}\n")
        reduce_image_features_wsi(
            data_paths=data_paths,
            global_dim=global_dim,
            local_dim=local_dim,
            scale=scale,
            delete_raw_features=delete_raw_features
        )
    print(f"\n{'#' * 80}")
    print(f"COMPLETED! Processed all scales: {scales}")
    print(f"{'#' * 80}")


def reduce_image_features_roi_wrapper(
    data_paths: List[str],
    global_dim: int,
    local_dim: int,
    scales: List[int],
    delete_raw_features: bool = False
) -> None:
    """
    Wrapper function to process ROI embeddings for multiple scales.

    Args:
        data_path: List of paths containing ROI embedding files
        global_dim: Target dimension for global features
        local_dim: Target dimension for local features
        scales: List of scale factors for embeddings
        delete_raw_features: If True and scale != 1, delete raw embedding files after processing
    """
    if isinstance(scales, int):
        scales = [scales]

    for scale in scales:
        print(f"\n{'#' * 80}")
        print(f"Processing ROI embeddings for scale: {scale}")
        print(f"{'#' * 80}\n")
        reduce_image_features_roi(
            data_paths=data_paths,
            global_dim=global_dim,
            local_dim=local_dim,
            scale=scale,
            delete_raw_features=delete_raw_features
        )

    print(f"\n{'#' * 80}")
    print(f"COMPLETED! Processed all scales: {scales}")
    print(f"{'#' * 80}")


def find_roi_indices(img_path: str, scale: int) -> List[int]:
    """
    Find all ROI indices i in embeddings_image_roi_i_scale_j.npy files for a specific scale.

    Args:
        img_path: Path containing ROI embedding files
        scale: Scale factor

    Returns:
        List of unique ROI indices sorted in ascending order
    """
    roi_indices = set()
    for filename in os.listdir(img_path):
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
        raise FileNotFoundError(f"No ROI embedding files found for scale {scale} in {img_path}")

    return sorted(list(roi_indices))


def extract_masked_features(
    img_path: str,
    max_i: int,
    mask: np.ndarray,
    feature_type: str = 'global'
) -> np.ndarray:
    """
    Extract features for masked pixels only, processing one file at a time.

    Args:
        img_path: Path containing embedding files
        max_i: Maximum embedding file index
        mask: 2D boolean mask array of shape (W, L)
        feature_type: 'global' for first 1024 dims, 'local' for last 1024 dims

    Returns:
        2D array of shape (num_masked_pixels, 1024)
    """
    W, L = mask.shape
    total_pixels = W * L
    mask_flat = mask.flatten()  # Flatten to 1D for indexing
    num_masked_pixels = np.sum(mask_flat)

    # Determine which dimension range to extract
    if feature_type == 'global':
        dim_start, dim_end = 0, 1024
    elif feature_type == 'local':
        dim_start, dim_end = 1024, 2048
    else:
        raise ValueError("feature_type must be 'global' or 'local'")

    # Initialize output array
    masked_features = np.zeros((num_masked_pixels, 1024), dtype=np.float32)

    # Track position in output array
    output_idx = 0
    current_pixel_idx = 0  # Track current position in the full pixel sequence

    # Process each embedding file
    for i in range(max_i + 1):
        embedding_file = os.path.join(img_path, f"embeddings_part_{i}.npy")

        if not os.path.exists(embedding_file):
            print(f"Warning: {embedding_file} not found, skipping...")
            continue

        print(f"Loading {embedding_file}...")
        embeddings = np.load(embedding_file)  # Shape: (n, 2048)
        n = embeddings.shape[0]

        # Determine which pixels from this file should be kept
        # Pixels in this file correspond to indices [current_pixel_idx : current_pixel_idx + n]
        file_pixel_range = slice(current_pixel_idx, current_pixel_idx + n)

        # Get mask values for pixels in this file
        mask_for_file = mask_flat[file_pixel_range]

        # Find which rows to keep (where mask is True)
        rows_to_keep = np.where(mask_for_file)[0]

        if len(rows_to_keep) > 0:
            # Extract only the needed features
            selected_features = embeddings[rows_to_keep, dim_start:dim_end]

            # Store in output array
            num_to_add = len(rows_to_keep)
            masked_features[output_idx:output_idx + num_to_add] = selected_features
            output_idx += num_to_add

        # Update current pixel index
        current_pixel_idx += n

        # Free memory
        del embeddings

    print(f"Extracted {output_idx} masked pixels")

    return masked_features


def main():
    parser = argparse.ArgumentParser(
        description="Reduce image feature dimensions from 2048 to target dimensions using PCA.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--data_paths',
        nargs='+',
        type=str,
        help='One or more paths containing data files'
    )

    parser.add_argument(
        '--global-dim',
        type=int,
        default=100,
        help='Target dimension for global features (default: 100)'
    )

    parser.add_argument(
        '--local-dim',
        type=int,
        default=100,
        help='Target dimension for local features (default: 100)'
    )

    parser.add_argument(
        '--scale',
        type=int,
        nargs='+',
        default=[1],
        help='Scale factor(s) for embeddings. Accepts one or more integers. If 1, reads from embeddings_part_i.npy and mask-small.png. '
             'If not 1, reads from embeddings_image_scale_j.npy and mask-small_scale_j.png (default: [1])'
    )
    parser.add_argument(
        '--wsi',
        action='store_true',
        default=False,
        help='Process WSI embeddings by PCA reduction (default: False)'
    )
    parser.add_argument(
        '--roi',
        action='store_true',
        default=False,
        help='Process ROI embeddings using pre-fitted PCA encoders. Reads from embeddings_image_roi_i_scale_j.npy '
             'and mask-small_roi_i_scale_j.png files (default: False)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        default=False,
        help='Delete raw embedding files after processing (only for scale != 1). '
             'This removes embeddings_image_scale_j.npy or embeddings_image_roi_i_scale_j.npy files (default: False)'
    )

    args = parser.parse_args()

    # Validate dimensions
    if args.global_dim <= 0 or args.local_dim <= 0:
        print("Error: Dimensions must be positive integers")
        return 1

    if args.global_dim > 1024:
        print("Warning: global_dim > 1024 (original dimension). Setting to 1024.")
        args.global_dim = 1024

    if args.local_dim > 1024:
        print("Warning: local_dim > 1024 (original dimension). Setting to 1024.")
        args.local_dim = 1024

    # Run dimension reduction
    print(f"\nConfiguration:")
    print(f"  Data paths: {args.data_paths}")
    print(f"  Global dimension: {args.global_dim}")
    print(f"  Local dimension: {args.local_dim}")
    print(f"  Total output dimension: {args.global_dim + args.local_dim}")
    print(f"  Scales: {args.scale}")
    print(f"  ROI mode: {args.roi}")
    print(f"  Clean (delete raw features): {args.clean}\n")
    
 
    if args.wsi:
        reduce_image_features_wsi_wrapper(
            data_paths=args.data_paths,
            global_dim=args.global_dim,
            local_dim=args.local_dim,
            scales=args.scale,
            delete_raw_features=args.clean
        )
    if args.roi:
        reduce_image_features_roi_wrapper(
            data_paths=args.data_paths,
            global_dim=args.global_dim,
            local_dim=args.local_dim,
            scales=args.scale,
            delete_raw_features=args.clean
        )

if __name__ == "__main__":
    sys.exit(main())
