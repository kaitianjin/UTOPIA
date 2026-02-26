from ..utils.io import load_mask, load_pickle
from ..utils.set_seeds import set_seeds
from .model_ct import FeedForwardClassifier, predict, train_model
import torch
import numpy as np
import os
import argparse
import torch.multiprocessing as mp
from typing import List, Union
from tqdm import tqdm


def train_ct_predictor(x, y, output_dim, 
                       n_epochs=20, 
                       batch_size=8096, 
                       learning_rate=1e-3, 
                       weight_decay=0,
                       dropout_rate=0.3,
                       temperature=1.0,
                       seed=42):
    set_seeds(seed)    
    model = FeedForwardClassifier(input_dim=x.shape[-1], 
                                  output_dim=output_dim, 
                                  dropout_rate=dropout_rate,
                                  temperature=temperature)
    history = train_model(model,
                          x,
                          y,
                          batch_size=batch_size,
                          epochs=n_epochs,
                          learning_rate=learning_rate,
                          weight_decay=weight_decay)
    return model

def infer_ct_predictor(model, x, batch_size=32):
    with torch.no_grad():
        yhat = predict(model, x, batch_size=batch_size)
    return yhat 

def train_fold_worker(train_x, train_y, calib_x, unique_cell_type_len, result_queue, fold_idx):
    if len(calib_x) == 0:
        print(f"Skipping fold {fold_idx} due to no calibration data.")
        result_queue.put((fold_idx, None))
        return

    model = train_ct_predictor(train_x, train_y, output_dim=unique_cell_type_len)
    calib_yhat = infer_ct_predictor(model, calib_x)

    # Ensure CUDA operations are complete before putting result in queue
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Make a contiguous copy to avoid any shared memory issues
    calib_yhat = np.ascontiguousarray(calib_yhat)
    result_queue.put((fold_idx, calib_yhat))

def reshape_roi_embeddings_3d(roi_embeddings, mask):
    h, w = mask.shape
    d = roi_embeddings.shape[1]
    reshaped_embeddings = np.zeros((h, w, d), dtype=roi_embeddings.dtype)
    reshaped_embeddings[mask]=roi_embeddings
    return reshaped_embeddings

def get_train_mask_for_roi(data_path, roi_idx):
    roi_he_mask = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{roi_idx}_scale_1.png"))
    roi_ct_mask = load_mask(os.path.join(data_path, "mask", f"ct-mask-small_roi_{roi_idx}_scale_1.png"))
    return roi_he_mask & roi_ct_mask 

def get_calib_mask_for_roi(data_path, roi_idx):
    roi_he_mask = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{roi_idx}_scale_1.png"))
    return roi_he_mask

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
        roi_ys[roi_idx] = np.load(os.path.join(data_path, "cell type", f"ct_image_roi_{roi_idx}_scale_1.npy")).astype(np.int64)
        roi_fold_masks[roi_idx] = load_pickle(os.path.join(data_path, "calibration", f"roi_{roi_idx}_fold_masks.pickle"))
        roi_center_masks[roi_idx] = load_pickle(os.path.join(data_path, "calibration", f"roi_{roi_idx}_center_masks.pickle"))
    return roi_xs, roi_ys, roi_train_masks, roi_calib_masks, roi_fold_masks, roi_center_masks

def cv_fold_calibration(roi_embed,
                        ct_image,
                        train_x_outside_cur_roi,
                        train_y_outside_cur_roi,
                        fold_masks,
                        center_masks,
                        train_mask_cur_roi,
                        calib_mask_cur_roi,
                        n_cell_types,
                        parallel_processes=8):
    n_folds = len(fold_masks)
    mp.set_start_method('spawn', force=True)

    W, L = roi_embed.shape[:2]
    calib_yhat_image = np.zeros((W, L, n_cell_types), dtype=np.float32)

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
            train_y = ct_image[train_mask]
            train_x = np.concatenate([train_x, train_x_outside_cur_roi], axis=0)
            train_y = np.concatenate([train_y, train_y_outside_cur_roi], axis=0)
            calib_x = roi_embed[calib_mask]
            p = mp.Process(
                target=train_fold_worker,
                args=(
                    train_x,
                    train_y,
                    calib_x,
                    n_cell_types,
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
    

def run(data_path: str,
        roi_train_indices: Union[int, List[int]],
        roi_calib_indices: Union[int, List[int]],
        n_cell_types: int,
        parallel_processes: int=8):

    if isinstance(roi_train_indices, int):
        roi_train_indices = [roi_train_indices]
    if isinstance(roi_calib_indices, int):
        roi_calib_indices = [roi_calib_indices]

    roi_indices = list(set(roi_train_indices+roi_calib_indices))
    roi_xs, roi_ys, roi_train_masks, roi_calib_masks, roi_fold_masks, roi_center_masks = get_roi_data(data_path, roi_indices)

    # train full model
    all_train_x = [roi_xs[i][roi_train_masks[i]] for i in roi_train_indices]
    all_train_y = [roi_ys[i][roi_train_masks[i]] for i in roi_train_indices]
    all_train_x = np.concatenate(all_train_x)
    all_train_y = np.concatenate(all_train_y)
    full_model = train_ct_predictor(all_train_x, all_train_y, output_dim=n_cell_types)
    # infer on wsi 
    wsi_x = np.load(os.path.join(data_path, "embeddings", "embeddings_scale_1.npy"))
    wsi_yhat = infer_ct_predictor(full_model, wsi_x)
    os.makedirs(os.path.join(data_path, "calibration", "cell type"), exist_ok=True)
    np.save(os.path.join(data_path, "calibration", "cell type", "ct_calib_scale_1.npy"), wsi_yhat)

    # build calibration data for each roi in roi_calib_indices
    for roi_idx in roi_calib_indices:
        print(f"Generating calibration predictions for ROI {roi_idx}...")
        if roi_idx not in roi_train_indices:
            calib_x = roi_xs[roi_idx][roi_calib_masks[roi_idx]]
            calib_yhat = infer_ct_predictor(full_model, calib_x)
        
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
                train_y_outside_cur_roi = np.empty((0,), dtype=roi_ys[roi_idx].dtype)
            roi_embed_cur_roi = roi_xs[roi_idx] 
            ct_image_cur_roi = roi_ys[roi_idx]
            fold_masks_cur_roi = roi_fold_masks[roi_idx]
            center_masks_cur_roi = roi_center_masks[roi_idx]
            calib_yhat = cv_fold_calibration(roi_embed_cur_roi,
                                             ct_image_cur_roi,
                                             train_x_outside_cur_roi,
                                             train_y_outside_cur_roi,
                                             fold_masks_cur_roi,
                                             center_masks_cur_roi,
                                             roi_train_masks[roi_idx],
                                             roi_calib_masks[roi_idx],
                                             n_cell_types,
                                             parallel_processes=parallel_processes)
        
        np.save(os.path.join(data_path, "calibration", "cell type", f"ct_calib_roi_{roi_idx}_scale_1.npy"), calib_yhat)


def main():
    parser = argparse.ArgumentParser(description="Train and calibrate cell type predictor")
    parser.add_argument('--data_path', type=str,
                        help='Path to the data directory')
    parser.add_argument('--roi_train_indices', type=int, nargs='+', required=True,
                        help='ROI indices for training (single integer or space-separated list, e.g., 0 or 0 1 2)')
    parser.add_argument('--roi_calib_indices', type=int, nargs='+', required=True,
                        help='ROI indices for calibration (single integer or space-separated list, e.g., 0 or 0 1 2)')
    parser.add_argument('--n_cell_types', type=int, required=True,
                        help='Number of cell types')
    parser.add_argument('--parallel_processes', type=int, default=8,
                        help='Number of parallel processes (default: 8)')

    args = parser.parse_args()

    run(data_path=args.data_path,
        roi_train_indices=args.roi_train_indices,
        roi_calib_indices=args.roi_calib_indices,
        n_cell_types=args.n_cell_types,
        parallel_processes=args.parallel_processes)


if __name__ == '__main__':
    main()