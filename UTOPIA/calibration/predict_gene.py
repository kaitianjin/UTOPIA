
from ..utils.io import load_pickle, save_pickle, load_mask
from ..utils.set_seeds import set_seeds
from .model_gene import FeedForwardNN, predict, train_model

import torch
import numpy as np
import os
import argparse
import torch.multiprocessing as mp
from typing import List, Union, Tuple
from tqdm import tqdm
from tqdm import tqdm


def train_gene_predictor(x, 
                         y,
                         n_epochs=20, 
                         batch_size=8096, 
                         learning_rate=1e-3,
                         beta=2,
                         loss='amse',
                         seed=42):
    set_seeds(seed)    
    model = FeedForwardNN(input_dim=x.shape[-1], output_dim=y.shape[-1])
    history = train_model(model, 
                          x, y, 
                          batch_size=batch_size, 
                          epochs=n_epochs,
                          learning_rate=learning_rate,
                          beta=beta,
                          loss=loss)
    return model

def infer_gene_predictor(model, x, batch_size=32):
    with torch.no_grad():
        yhat = predict(model, x, batch_size=batch_size)
    return yhat 

def reshape_roi_embeddings_3d(roi_embeddings, mask):
    h, w = mask.shape
    d = roi_embeddings.shape[1]
    reshaped_embeddings = np.zeros((h, w, d), dtype=roi_embeddings.dtype)
    reshaped_embeddings[mask]=roi_embeddings
    return reshaped_embeddings

def get_train_mask_for_roi(data_path, roi_idx):
    roi_he_mask = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{roi_idx}_scale_1.png"))
    return roi_he_mask

def get_calib_mask_for_roi(data_path, roi_idx):
    roi_he_mask = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{roi_idx}_scale_1.png"))
    return roi_he_mask

def rescaling_gene_image(
    roi_ys: List[np.ndarray],
    roi_train_masks: List[np.ndarray],
    upper_quantile: float = 99.99,
    lower_quantile: float = 0,
    idx_to_gene: Union[dict, None] = None
) -> Tuple[List[np.ndarray], dict]:
    """
    Standardize gene expression counts to values between 0 and 1.

    Args:
        roi_ys: List of 3D arrays (W, L, C) with gene counts (uint16)
        roi_train_masks: List of 2D boolean masks (W, L)
        upper_quantile: Quantile for max threshold (0-100), default 99.99
        lower_quantile: Quantile for min threshold (0-100), default 0
        idx_to_gene: Optional dictionary mapping gene indices to gene names
    Returns:
        roi_ys_std: List of 3D float32 arrays with standardized values in [0, 1]
        gene_thresholds: Dict mapping gene index or name to (lower_threshold, upper_threshold)
    """
    num_roi = len(roi_ys)
    num_genes = roi_ys[0].shape[2]

    roi_ys_std = [np.zeros(roi_ys[i].shape, dtype=np.float32) for i in range(num_roi)]
    gene_thresholds = {}

    for gene_idx in tqdm(range(num_genes), desc="Rescaling gene counts", unit='genes'):
        gene_counts = []
        for i in range(num_roi):
            gene_counts.append(roi_ys[i][roi_train_masks[i], gene_idx])
        gene_counts = np.concatenate(gene_counts)

        lower_threshold = np.percentile(gene_counts, lower_quantile)
        upper_threshold = np.percentile(gene_counts, upper_quantile)
        if idx_to_gene is not None:
            gene_name = idx_to_gene[gene_idx]
            gene_thresholds[gene_name] = (lower_threshold, upper_threshold)
        else:
            gene_thresholds[gene_idx] = (lower_threshold, upper_threshold)

        for i in range(num_roi):
            if upper_threshold != lower_threshold:
                clipped = np.clip(roi_ys[i][..., gene_idx], lower_threshold, upper_threshold)
                scaled = (clipped - lower_threshold) / (upper_threshold - lower_threshold)
                roi_ys_std[i][roi_train_masks[i], gene_idx] = scaled[roi_train_masks[i]]
            # Where roi_train_masks[i] is False, values remain 0 (initialized)

    return roi_ys_std, gene_thresholds

def get_roi_data(data_path, roi_indices):
    num_roi = len(roi_indices)
    roi_xs = [None for _ in range(num_roi)]
    roi_train_masks = [None for _ in range(num_roi)]
    roi_calib_masks = [None for _ in range(num_roi)]
    roi_ys = [None for _ in range(num_roi)]
    roi_fold_masks = [None for _ in range(num_roi)]
    roi_center_masks = [None for _ in range(num_roi)]
    for roi_idx in roi_indices:
        roi_embeddings = np.load(os.path.join(data_path, "embeddings", f"embeddings_roi_{roi_idx}_scale_1.npy"))
        roi_train_masks[roi_idx] = get_train_mask_for_roi(data_path, roi_idx)
        roi_calib_masks[roi_idx] = get_calib_mask_for_roi(data_path, roi_idx)
        roi_xs[roi_idx] = reshape_roi_embeddings_3d(roi_embeddings, roi_calib_masks[roi_idx])
        roi_ys[roi_idx] = np.load(os.path.join(data_path, "transcripts", f"tr_image_roi_{roi_idx}_scale_1.npy")).astype(np.uint16)
        roi_fold_masks[roi_idx] = load_pickle(os.path.join(data_path, "calibration", f"roi_{roi_idx}_fold_masks.pickle"))
        roi_center_masks[roi_idx] = load_pickle(os.path.join(data_path, "calibration", f"roi_{roi_idx}_center_masks.pickle"))
    return roi_xs, roi_ys, roi_train_masks, roi_calib_masks, roi_fold_masks, roi_center_masks

def train_fold_worker(train_x, 
                      train_y, 
                      calib_x, 
                      result_queue, 
                      fold_idx):
    if len(calib_x) == 0:
        print(f"Skipping fold {fold_idx} due to no calibration data.")
        result_queue.put((fold_idx, None))
        return

    model = train_gene_predictor(train_x, train_y)
    calib_yhat = infer_gene_predictor(model, calib_x)

    # Ensure CUDA operations are complete before putting result in queue
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Make a contiguous copy to avoid any shared memory issues
    calib_yhat = np.ascontiguousarray(calib_yhat)
    result_queue.put((fold_idx, calib_yhat))

def cv_fold_calibration(roi_embed,
                        tr_image,
                        train_x_outside_cur_roi,
                        train_y_outside_cur_roi,
                        fold_masks,
                        center_masks,
                        train_mask_cur_roi,
                        calib_mask_cur_roi,
                        parallel_processes=8):
    n_folds = len(fold_masks)
    mp.set_start_method('spawn', force=True)

    W, L = roi_embed.shape[:2]
    n_genes = tr_image.shape[2]
    calib_yhat_image = np.zeros((W, L, n_genes), dtype=np.float32)

    result_queue = mp.Queue()
    calib_masks_per_fold = {}

    fold_progress = tqdm(total=n_folds,
                         desc="Cross-Fold Training",
                         ncols=80,
                         colour="green",
                         unit="fold",
                         leave=True)
    for batch_start in range(0, n_folds, parallel_processes):
        processes = []
        batch_end = min(batch_start + parallel_processes, n_folds)
        for fold_idx in range(batch_start, batch_end):
            train_mask = (~fold_masks[fold_idx]) & train_mask_cur_roi
            calib_mask = center_masks[fold_idx] & calib_mask_cur_roi
            calib_masks_per_fold[fold_idx] = calib_mask
            train_x = roi_embed[train_mask]
            train_y = tr_image[train_mask]
            train_x = np.concatenate([train_x, train_x_outside_cur_roi], axis=0)
            train_y = np.concatenate([train_y, train_y_outside_cur_roi], axis=0)
            calib_x = roi_embed[calib_mask]
            p = mp.Process(
                target=train_fold_worker,
                args=(
                    train_x,
                    train_y,
                    calib_x,
                    result_queue,
                    fold_idx,
                )
            )
            p.start()
            processes.append(p)

        # IMPORTANT: Get from queue BEFORE joining processes to avoid deadlock
        # If a process fills the queue buffer, it will block on put() until
        # the main process reads from the queue. If we join() first, we deadlock.
        for _ in range(len(processes)):
            fold_idx, calib_yhat = result_queue.get()
            if calib_yhat is not None:
                calib_yhat_image[calib_masks_per_fold[fold_idx]] = calib_yhat
            fold_progress.update(1)

        for p in processes:
            p.join()

    fold_progress.close()
    return calib_yhat_image[calib_mask_cur_roi]

def load_index_to_gene(data_path: str) -> dict:
    if not os.path.exists(os.path.join(data_path, "transcripts", "idx_to_gene.pickle")):
        return None
    idx_to_gene = load_pickle(os.path.join(data_path, "transcripts", "idx_to_gene.pickle"))
    return idx_to_gene

def run(data_path: str,
        roi_train_indices: Union[int, List[int]],
        roi_calib_indices: Union[int, List[int]],
        parallel_processes: int=8):

    if isinstance(roi_train_indices, int):
        roi_train_indices = [roi_train_indices]
    if isinstance(roi_calib_indices, int):
        roi_calib_indices = [roi_calib_indices]

    idx_to_gene = load_index_to_gene(data_path)

    roi_indices = list(set(roi_train_indices+roi_calib_indices))
    roi_xs, roi_ys, roi_train_masks, roi_calib_masks, roi_fold_masks, roi_center_masks = get_roi_data(data_path, roi_indices)
    roi_ys, gene_thresholds = rescaling_gene_image(roi_ys, roi_train_masks, idx_to_gene=idx_to_gene)
    os.makedirs(os.path.join(data_path, "calibration", "gene"), exist_ok=True)
    save_pickle(gene_thresholds, os.path.join(data_path, "calibration", "gene", "gene_thresholds.pickle"))

    # train full model
    all_train_x = [roi_xs[i][roi_train_masks[i]] for i in roi_train_indices]
    all_train_y = [roi_ys[i][roi_train_masks[i]] for i in roi_train_indices]
    all_train_x = np.concatenate(all_train_x)
    all_train_y = np.concatenate(all_train_y)
    full_model = train_gene_predictor(all_train_x, all_train_y)
    # infer on wsi 
    wsi_x = np.load(os.path.join(data_path,"embeddings", "embeddings_scale_1.npy"))
    wsi_yhat = infer_gene_predictor(full_model, wsi_x)
    np.save(os.path.join(data_path, "calibration", "gene", "gene_calib_scale_1.npy"), wsi_yhat)

    # build calibration data for each roi in roi_calib_indices
    for roi_idx in roi_calib_indices:
        print(f"Generating calibration predictions for ROI {roi_idx}...")
        if roi_idx not in roi_train_indices:
            calib_x = roi_xs[roi_idx][roi_calib_masks[roi_idx]]
            calib_yhat = infer_gene_predictor(full_model, calib_x)
        
        if roi_idx in roi_train_indices:
            train_x_outside_cur_roi = []
            train_y_outside_cur_roi = []
            for other_roi_idx in roi_train_indices:
                if other_roi_idx != roi_idx:
                    train_x_outside_cur_roi.append(roi_xs[other_roi_idx][roi_train_masks[other_roi_idx]])
                    train_y_outside_cur_roi.append(roi_ys[other_roi_idx][roi_train_masks[other_roi_idx]])
            if train_x_outside_cur_roi:
                train_x_outside_cur_roi = np.concatenate(train_x_outside_cur_roi, axis=0)
                train_y_outside_cur_roi = np.concatenate(train_y_outside_cur_roi, axis=0)
            else:
                train_x_outside_cur_roi = np.empty((0, roi_xs[roi_idx].shape[-1]), dtype=roi_xs[roi_idx].dtype)
                train_y_outside_cur_roi = np.empty((0, roi_ys[roi_idx].shape[-1]), dtype=roi_ys[roi_idx].dtype)
            roi_embed_cur_roi = roi_xs[roi_idx] 
            tr_image_cur_roi = roi_ys[roi_idx]
            fold_masks_cur_roi = roi_fold_masks[roi_idx]
            center_masks_cur_roi = roi_center_masks[roi_idx]
            calib_yhat = cv_fold_calibration(roi_embed_cur_roi,
                                             tr_image_cur_roi,
                                             train_x_outside_cur_roi,
                                             train_y_outside_cur_roi,
                                             fold_masks_cur_roi,
                                             center_masks_cur_roi,
                                             roi_train_masks[roi_idx],
                                             roi_calib_masks[roi_idx],
                                             parallel_processes=parallel_processes)
        
        np.save(os.path.join(data_path, "calibration", "gene", f"gene_calib_roi_{roi_idx}_scale_1.npy"), calib_yhat)


def main():
    parser = argparse.ArgumentParser(description="Train and calibrate cell type predictor")
    parser.add_argument('--data_path', type=str,
                        help='Path to the image directory')
    parser.add_argument('--roi_train_indices', type=int, nargs='+', required=True,
                        help='ROI indices for training (single integer or space-separated list, e.g., 0 or 0 1 2)')
    parser.add_argument('--roi_calib_indices', type=int, nargs='+', required=True,
                        help='ROI indices for calibration (single integer or space-separated list, e.g., 0 or 0 1 2)')
    parser.add_argument('--parallel_processes', type=int, default=8,
                        help='Number of parallel processes (default: 8)')

    args = parser.parse_args()

    run(data_path=args.data_path,
        roi_train_indices=args.roi_train_indices,
        roi_calib_indices=args.roi_calib_indices,
        parallel_processes=args.parallel_processes)


if __name__ == '__main__':
    main()
        
