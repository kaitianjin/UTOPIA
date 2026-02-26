from typing import List, Union
import numpy as np
import os 
from ..utils.io import save_pickle, load_mask

def find_rectangle(mask):
    true_indices = np.where(mask)
    
    min_row = np.min(true_indices[0])
    max_row = np.max(true_indices[0])
    min_col = np.min(true_indices[1])
    max_col = np.max(true_indices[1])
    
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    
    expected_true_count = width * height
    actual_true_count = len(true_indices[0])
    
    assert expected_true_count == actual_true_count
    
    return (min_col, min_row), height, width

def generate_folds(mask, N, check_boundary, safe_boundary=16):

        
    ROI_top_left, ROI_height, ROI_width = find_rectangle(mask)
    W, L = mask.shape[1], mask.shape[0]
    
    # Extract ROI coordinates
    roi_x, roi_y = ROI_top_left
    roi_right = roi_x + ROI_width
    roi_bottom = roi_y + ROI_height

    check_left, check_right, check_top, check_bottom = check_boundary[0], check_boundary[1], check_boundary[2], check_boundary[3]

    if (ROI_width - safe_boundary*(1-check_left) - safe_boundary*(1-check_right))<=0:
        raise Exception("The mask is not wide enough.")
    if (ROI_height - safe_boundary*(1-check_bottom) - safe_boundary*(1-check_top))<=0:
        raise Exception("The mask is not high enough.") 
    # Initialize lists to store fold masks and center masks
    fold_masks = []
    center_masks = []
    check_boundaries = []
    # Step size for moving folds (with overlap)
    step_size = N - 2*safe_boundary
    
    # Generate folds
    for y in range(roi_y, roi_bottom, step_size):
        for x in range(roi_x, roi_right, step_size):
            # Create fold boundaries
            fold_left = x
            fold_top = y
            fold_right = min(x + N, roi_right)
            fold_bottom = min(y + N, roi_bottom)
            # Create mask for the current fold
            fold_mask = np.zeros((L, W), dtype=bool)
            fold_mask[fold_top:fold_bottom, fold_left:fold_right] = True
            
            # Create a mask for the center of the fold
            center_mask = np.zeros((L, W), dtype=bool)
            
            # Determine which boundaries touch the ROI
            touches_left = (fold_left == roi_x) & (check_left)
            touches_right = (fold_right == roi_right) & (check_right)
            touches_top = (fold_top == roi_y) & (check_top)
            touches_bottom = (fold_bottom == roi_bottom) & (check_bottom)
            # Define the center by excluding safe_boundary pixels from non-touching boundaries
            center_left = fold_left + (0 if touches_left else safe_boundary)
            center_right = fold_right - (0 if touches_right else safe_boundary)
            center_top = fold_top + (0 if touches_top else safe_boundary)
            center_bottom = fold_bottom - (0 if touches_bottom else safe_boundary)
            
            # Set the center area to True
            if center_right > center_left and center_bottom > center_top:
                center_mask[center_top:center_bottom, center_left:center_right] = True
                
                # Add fold mask and center mask to the lists
                fold_masks.append(fold_mask)
                center_masks.append(center_mask)
                check_boundaries.append([touches_left, touches_right, touches_top, touches_bottom])
            if center_right == roi_right: 
                break 
        if center_bottom == roi_bottom:
            break
    return fold_masks, center_masks, check_boundaries

def count_clusters(he_clusters_image, clusters_to_count):
    clusters, counts = np.unique(he_clusters_image, return_counts=True)
    full_count = np.full(len(clusters_to_count), 0, int)
    for i, cluster in enumerate(clusters_to_count):
        if cluster in clusters:
            full_count[i] = counts[clusters==cluster].item() 
        else:
            full_count[i] = 0
    return full_count

def split_into_folds(observed_mask, 
                     fold_mask, center_mask, 
                     final_fold_masks, final_center_masks, 
                     he_clusters_image, clusters_in_ROI, 
                     fold_widths, safe_boundary, 
                     iteration,
                     check_boundary):
    
    cannot_countinue = False
    try:
        small_fold_masks, small_center_masks, small_check_boundaries = generate_folds(fold_mask, fold_widths[iteration], check_boundary, safe_boundary)
    except Exception: 
        cannot_countinue = True 

    
    if iteration > 0:
        clusters_counts_in_train = count_clusters(he_clusters_image[observed_mask & (~fold_mask)], clusters_in_ROI)
        clusters_counts_in_ROI = count_clusters(he_clusters_image[observed_mask], clusters_in_ROI)
        clusters_proportion = clusters_counts_in_train/clusters_counts_in_ROI

        if np.all(clusters_proportion>=0.8) or iteration==len(fold_widths) or cannot_countinue:
            final_fold_masks.append(fold_mask)
            final_center_masks.append(center_mask) 
            return None 
        
    for i in range(len(small_fold_masks)):
        split_into_folds(observed_mask,
                         small_fold_masks[i],small_center_masks[i], 
                         final_fold_masks, final_center_masks, 
                         he_clusters_image, clusters_in_ROI, 
                         fold_widths, safe_boundary, 
                         iteration+1,
                         small_check_boundaries[i])

            
    return None
    
def generate_folds_final(he_clusters_image, observed_mask, ROI_mask, fold_widths, safe_boundary):


    clusters_in_ROI = np.unique(he_clusters_image[observed_mask])
    final_fold_masks = []
    final_center_masks = []
    split_into_folds(observed_mask,
                     ROI_mask,ROI_mask, 
                     final_fold_masks, final_center_masks, 
                     he_clusters_image, clusters_in_ROI,
                     fold_widths, safe_boundary,
                     check_boundary=[True, True, True, True],
                     iteration=0)
    return final_fold_masks, final_center_masks

def find_roi_indices(path: str, scale: int) -> List[int]:
    """
    Find all ROI indices i in embeddings_image_roi_i_scale_j.npy files for a specific scale.

    Args:
        
        path: Path containing ROI embedding files
        scale: Scale factor

    Returns:
        List of unique ROI indices sorted in ascending order
    """
    roi_indices = set()
    for filename in os.listdir(path):
        if filename.startswith(f"he_clusters_image_roi_"):
            try:
                parts = filename.replace("he_clusters_image_roi_", "").split(f"_scale_")
                roi_idx = int(parts[0])
                roi_indices.add(roi_idx)
            except (ValueError, IndexError):
                continue

    return sorted(list(roi_indices))

def generate_one_fold_per_roi(data_path: str):
    """
    Generate a single fold per ROI that covers the entire ROI.
    Both fold_mask and center_mask are boolean arrays with all True values.

    Args:
        img_path: Path to the image directory
    """
    roi_indices = find_roi_indices(os.path.join(data_path, "histology_clustering"), scale=1)

    for i in roi_indices:
        roi_reserve_index_image = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{i}_scale_1.png"))
        full_mask = np.ones(roi_reserve_index_image.shape, dtype=bool)
        fold_masks = [full_mask]
        center_masks = [full_mask]
        os.makedirs(os.path.join(data_path, "calibration"), exist_ok=True)
        save_pickle(fold_masks, os.path.join(data_path, "calibration", f"roi_{i}_fold_masks.pickle"))
        save_pickle(center_masks, os.path.join(data_path, "calibration", f"roi_{i}_center_masks.pickle"))

def generate_cross_validation_folds(
        data_path: str,
        n_clusters: int,
        fold_widths: Union[int, List[int]]=[200, 100, 50],
        safe_boundary: int=16,
):
    if isinstance(fold_widths, int):
        fold_widths = [fold_widths]
    clusters_counts_in_all_ROIs = [0 for _ in range(n_clusters)]
    roi_indices = find_roi_indices(os.path.join(data_path, "histology_clustering"), scale=1)

    for i in roi_indices:
        roi_he_clusters_image = np.load(os.path.join(data_path, "histology_clustering", f"he_clusters_image_roi_{i}_scale_1_num_{n_clusters}.npy"))
        available_clusters, counts = np.unique(roi_he_clusters_image,return_counts=True) 
        for cluster in range(n_clusters): 
            if cluster in available_clusters:
                clusters_counts_in_all_ROIs[cluster] += counts[np.where(available_clusters==cluster)].item()
    for i in roi_indices:
        roi_he_clusters_image = np.load(os.path.join(data_path, "histology_clustering", f"he_clusters_image_roi_{i}_scale_1_num_{n_clusters}.npy"))
        roi_reserve_index_image = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{i}_scale_1.png"))
        fold_masks, center_masks = generate_folds_final(roi_he_clusters_image,
                     roi_reserve_index_image,
                     np.ones(roi_reserve_index_image.shape,dtype=bool),
                     fold_widths=fold_widths,
                     safe_boundary=safe_boundary)
        os.makedirs(os.path.join(data_path, "calibration"), exist_ok=True)
        save_pickle(fold_masks, os.path.join(data_path, "calibration", f"roi_{i}_fold_masks.pickle"))
        save_pickle(center_masks, os.path.join(data_path, "calibration", f"roi_{i}_center_masks.pickle"))

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cross-validation folds for calibration")
    parser.add_argument("data_path", type=str, help="Path to the data directory")
    parser.add_argument("--n_clusters", type=int, required=True, help="Number of clusters")
    parser.add_argument("--fold_widths", type=int, nargs="+", default=[200, 100, 50],
                        help="List of fold widths or a single integer (default: 200 100 50)")
    parser.add_argument("--safe_boundary", type=int, default=16,
                        help="Safe boundary size (default: 16)")
    parser.add_argument("--one_fold_per_roi", action="store_true",
                        help="Generate only one fold per ROI covering the entire ROI")

    args = parser.parse_args()

    if args.one_fold_per_roi:
        generate_one_fold_per_roi(data_path=args.data_path)
    else:
        generate_cross_validation_folds(
            data_path=args.data_path,
            n_clusters=args.n_clusters,
            fold_widths=args.fold_widths,
            safe_boundary=args.safe_boundary
        )


if __name__ == "__main__":
    main()
