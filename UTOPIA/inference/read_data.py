from collections import defaultdict
import os
import numpy as np
import json
from ..utils.sliding_window import sliding_window
from ..utils.io import load_pickle, load_mask 

def load_main_data(data_path, 
                   num_he_clusters, 
                   scales, 
                   roi_calib_indices):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    roi_mask = np.load(os.path.join(data_path, "mask", "roi_mask_scale_1.npy"))
    out_roi_mask = ~roi_mask 

    for scale in scales:
        data['WSI']['reserve_index_image'][scale] = load_mask(os.path.join(data_path, "mask", f"mask-small_scale_{scale}.png"))
        data['WSI']['he_clusters_image'][scale] = np.load(os.path.join(data_path, "histology_clustering", f"he_clusters_image_scale_{scale}_num_{num_he_clusters}.npy")).astype(np.uint8)
        data['WSI']['roi_mask'][scale] = sliding_window(roi_mask, window_shape=(scale,scale),stride=(scale,scale)).astype(bool)
        data['WSI']['out_roi_mask'][scale] = sliding_window(out_roi_mask, window_shape=(scale,scale),stride=(scale,scale)).astype(bool)

    for i in roi_calib_indices:
        for scale in scales: 
            data['ROI']['he_clusters_images'][scale][i] = np.load(os.path.join(data_path, "histology_clustering", f"he_clusters_image_roi_{i}_scale_{scale}_num_{num_he_clusters}.npy")).astype(np.uint8)
            data['ROI']['reserve_index_images'][scale][i] = load_mask(os.path.join(data_path, "mask", f"mask-small_roi_{i}_scale_{scale}.png")).astype(bool)

    return data

def load_gene_data(data_path,
                   roi_calib_indices,
                   main_data):
    gene_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    idx_to_gene = load_pickle(os.path.join(data_path, "transcripts", "idx_to_gene.pickle"))
    gene_to_idx = {gene: idx for idx, gene in idx_to_gene.items()}
    gene_data['idx_to_gene'] = idx_to_gene 
    gene_data['gene_to_idx'] = gene_to_idx 
    gene_data['gene_thresholds'] = load_pickle(os.path.join(data_path, "calibration", "gene", "gene_thresholds.pickle"))

    scale = 1
    total_pred_3d_image = np.zeros((*main_data['WSI']['roi_mask'][scale].shape,len(gene_data['idx_to_gene'])),dtype=np.float32)
    total_pred_3d_image[main_data['WSI']['reserve_index_image'][scale]] = np.load(os.path.join(data_path, "calibration", "gene", "gene_calib_scale_1.npy"))
    gene_data['total_true_3d_image'][scale] = np.load(os.path.join(data_path, "transcripts", "tr_image_scale_1.npy"))
    gene_data['total_pred_3d_image'][scale] = total_pred_3d_image     

    for roi_idx in roi_calib_indices: 
        calib_pred_3d_image = np.zeros((*main_data['ROI']['reserve_index_images'][scale][roi_idx].shape,len(gene_data['idx_to_gene'])),dtype=np.float32)
        calib_pred_3d_image[main_data['ROI']['reserve_index_images'][scale][roi_idx]] = np.load(os.path.join(data_path, "calibration", "gene", f"gene_calib_roi_{roi_idx}_scale_{scale}.npy"))
        calib_true_3d_image = np.load(os.path.join(data_path, "transcripts", f"tr_image_roi_{roi_idx}_scale_{scale}.npy"))
        gene_data['calib_pred_3d_images'][scale][roi_idx] = calib_pred_3d_image
        gene_data['calib_true_3d_images'][scale][roi_idx] = calib_true_3d_image

    return gene_data

def load_ct_data(data_path,
                 roi_calib_indices,
                 main_data,
                 scales):
    ct_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    idx_to_ct = load_pickle(os.path.join(data_path, "cell type", "idx_to_ct.pickle"))
    ct_to_idx = {ct: idx for idx, ct in idx_to_ct.items()}
    ct_data['idx_to_ct'] = idx_to_ct 
    ct_data['ct_to_idx'] = ct_to_idx 

    scale = 1
    total_pred_3d_image = np.zeros((*main_data['WSI']['roi_mask'][scale].shape,len(ct_data['idx_to_ct'])),dtype=np.float32)
    total_pred_3d_image[main_data['WSI']['reserve_index_image'][scale]] = np.load(os.path.join(data_path, "calibration", "cell type", "ct_calib_scale_1.npy"))
    ct_data['total_pred_3d_image'][scale] = total_pred_3d_image     

    ct_data['ct_image'] = np.load(os.path.join(data_path, "cell type", "ct_image_scale_1.npy"))
    total_true_3d_image = np.zeros_like(total_pred_3d_image, dtype=np.float32)
    for i in ct_data['idx_to_ct']:
        total_true_3d_image[ct_data['ct_image']==i,i]=1
    ct_data['total_true_3d_image'][scale] = total_true_3d_image
    
    for roi_idx in roi_calib_indices:
        scale = 1
        calib_pred_3d_image = np.zeros((*main_data['ROI']['reserve_index_images'][scale][roi_idx].shape,len(ct_data['idx_to_ct'])),dtype=np.float32)
        calib_pred_3d_image[main_data['ROI']['reserve_index_images'][scale][roi_idx]] = np.load(os.path.join(data_path, "calibration", "cell type", f"ct_calib_roi_{roi_idx}_scale_{scale}.npy"))
        calib_true_3d_image = np.zeros_like(calib_pred_3d_image, dtype=np.float32)
        ct_image = np.load(os.path.join(data_path, "cell type", f"ct_image_roi_{roi_idx}_scale_{scale}.npy"))
        for i in ct_data['idx_to_ct']:
            calib_true_3d_image[ct_image==i,i]=1
        ct_data['calib_pred_3d_images'][scale][roi_idx] = calib_pred_3d_image
        ct_data['calib_true_3d_images'][scale][roi_idx] = calib_true_3d_image

    for scale in scales:
        ct_data['wsi_ct_mask'][scale] = load_mask(os.path.join(data_path, "mask", f"ct-mask-small_scale_{scale}.png")).astype(bool)

        for roi_idx in roi_calib_indices:
            ct_data['roi_ct_mask'][scale][roi_idx] = load_mask(os.path.join(data_path, "mask", f"ct-mask-small_roi_{roi_idx}_scale_{scale}.png")).astype(bool)
    
    return ct_data


def weight_sum(image_3d, weights): 
    output_image = np.zeros(image_3d.shape[:2]) 
    for g in range(image_3d.shape[2]): 
        output_image += image_3d[..., g] * weights[g]
    return output_image

def load_gene_calib_data(main_data,
                         gene_data,
                         genes,
                         roi_calib_indices,
                         scales):
    calib_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    gene_indices = []
    gene_weights = [] 
    for gene in genes: 
        gene_idx = gene_data['gene_to_idx'][gene]
        gene_indices.append(gene_idx) 
        gene_weights.append(gene_data['gene_thresholds'][gene][1])
    gene_indices = np.array(gene_indices)
    gene_weights = np.array(gene_weights)


    for scale in scales:
        calib_data['he_clusters'][scale]=[] 
        calib_data['calib_yhat'][scale]=[] 
        calib_data['calib_y'][scale]=[] 
        norm_factor = 1
        for roi_idx in roi_calib_indices:

            calib_data['calib_pred_images'][scale][roi_idx] = sliding_window(weight_sum(gene_data['calib_pred_3d_images'][1][roi_idx][...,gene_indices],
                                                                                        weights=gene_weights),
                                                                             window_shape=(scale,scale),stride=(scale,scale), method='sum')
            calib_data['calib_true_images'][scale][roi_idx] = sliding_window(np.sum(gene_data['calib_true_3d_images'][1][roi_idx][...,gene_indices],axis=-1),
                                                                             window_shape=(scale,scale),stride=(scale,scale), method='sum')

            calib_mask = main_data['ROI']['reserve_index_images'][scale][roi_idx] 
            gene_counts_in_true = calib_data['calib_true_images'][scale][roi_idx][calib_mask]
            norm_factor = max(norm_factor, np.percentile(gene_counts_in_true,99.99))

            calib_data['calib_true_images'][scale][roi_idx] = calib_data['calib_true_images'][scale][roi_idx].astype(np.float32)
            calib_data['calib_true_images'][scale][roi_idx] /= norm_factor 
            calib_data['calib_pred_images'][scale][roi_idx] /= norm_factor 
            calib_data['calib_yhat'][scale].append(calib_data['calib_pred_images'][scale][roi_idx][calib_mask])
            calib_data['calib_y'][scale].append(calib_data['calib_true_images'][scale][roi_idx][calib_mask])
            calib_data['he_clusters'][scale].append(main_data['ROI']['he_clusters_images'][scale][roi_idx][calib_mask])

        calib_data['he_clusters'][scale]=np.concatenate(calib_data['he_clusters'][scale])
        calib_data['calib_yhat'][scale]=np.concatenate(calib_data['calib_yhat'][scale]) 
        calib_data['calib_y'][scale]=np.concatenate(calib_data['calib_y'][scale]) 
        calib_data['calib_max'][scale] = norm_factor

    return calib_data 

def load_gene_test_data(main_data,
                        gene_data,
                        calib_data,
                        genes,
                        scales):
    test_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    gene_indices = []
    gene_weights = []
    for gene in genes: 
        gene_idx = gene_data['gene_to_idx'][gene]
        gene_indices.append(gene_idx) 
        gene_weights.append(gene_data['gene_thresholds'][gene][1])
    gene_indices = np.array(gene_indices)
    gene_weights = np.array(gene_weights)

    for scale in scales:
        test_data['he_clusters'][scale]=[] 
        test_data['test_yhat'][scale]=[] 
        test_data['test_y'][scale]=[] 

        test_mask = main_data['WSI']['reserve_index_image'][scale] & main_data['WSI']['out_roi_mask'][scale]
        test_data['test_mask'][scale] = test_mask

        test_data['total_pred_image'][scale] = sliding_window(weight_sum(gene_data['total_pred_3d_image'][1][...,gene_indices],
                                                                         weights=gene_weights),
                                                            window_shape=(scale,scale),stride=(scale,scale), method='sum')
        test_data['total_true_image'][scale] = sliding_window(np.sum(gene_data['total_true_3d_image'][1][...,gene_indices],axis=-1),
                                                            window_shape=(scale,scale),stride=(scale,scale), method='sum')
        
        test_data['total_true_image'][scale] = test_data['total_true_image'][scale].astype(np.float32) 
        test_data['total_true_image'][scale] /= calib_data['calib_max'][scale]
        test_data['total_pred_image'][scale] /= calib_data['calib_max'][scale]
        
        test_data['fdr_mask'][scale] = test_mask
        test_data['test_yhat'][scale]=(test_data['total_pred_image'][scale][test_mask])
        test_data['test_y'][scale]=(test_data['total_true_image'][scale][test_mask])
        test_data['he_clusters'][scale]=(main_data['WSI']['he_clusters_image'][scale][test_mask])

    return test_data

def load_ct_calib_data(main_data,
                       ct_data,
                       cell_types,
                       roi_calib_indices,
                       scales):  
    calib_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ct_indices = []
    for ct in cell_types: 
        ct_idx = ct_data['ct_to_idx'][ct]
        ct_indices.append(ct_idx) 
    ct_indices = np.array(ct_indices)

    for scale in scales:
        calib_data['he_clusters'][scale]=[] 
        calib_data['calib_yhat'][scale]=[] 
        calib_data['calib_y'][scale]=[] 
        for roi_idx in roi_calib_indices:

            calib_data['calib_pred_images'][scale][roi_idx] = sliding_window(np.sum(ct_data['calib_pred_3d_images'][1][roi_idx][...,ct_indices],
                                                                                    axis=-1),
                                                                             window_shape=(scale,scale),stride=(scale,scale), method='mean')
            calib_data['calib_true_images'][scale][roi_idx] = sliding_window(np.sum(ct_data['calib_true_3d_images'][1][roi_idx][...,ct_indices],
                                                                                    axis=-1),
                                                                             window_shape=(scale,scale),stride=(scale,scale), method='sum')

            calib_mask = main_data['ROI']['reserve_index_images'][scale][roi_idx] & ct_data['roi_ct_mask'][scale][roi_idx]

            calib_data['calib_yhat'][scale].append(calib_data['calib_pred_images'][scale][roi_idx][calib_mask])
            calib_data['calib_y'][scale].append(calib_data['calib_true_images'][scale][roi_idx][calib_mask])
            calib_data['he_clusters'][scale].append(main_data['ROI']['he_clusters_images'][scale][roi_idx][calib_mask])

        
        calib_data['he_clusters'][scale]=np.concatenate(calib_data['he_clusters'][scale])
        calib_data['calib_yhat'][scale]=np.concatenate(calib_data['calib_yhat'][scale]) 
        calib_data['calib_y'][scale]=np.concatenate(calib_data['calib_y'][scale]) 

    return calib_data 
    
def load_ct_test_data(main_data,
                        ct_data,
                        cell_types,
                        scales): 
    test_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ct_indices = []
    for ct in cell_types: 
        ct_idx = ct_data['ct_to_idx'][ct]
        ct_indices.append(ct_idx) 
    ct_indices = np.array(ct_indices)

    for scale in scales:
        test_data['he_clusters'][scale]=[] 
        test_data['test_yhat'][scale]=[] 
        test_data['test_y'][scale]=[] 


        test_data['total_pred_image'][scale] = sliding_window(np.sum(ct_data['total_pred_3d_image'][1][...,ct_indices],
                                                                                    axis=-1),
                                                                             window_shape=(scale,scale),stride=(scale,scale), method='mean')
        test_data['total_true_image'][scale] = sliding_window(np.sum(ct_data['total_true_3d_image'][1][...,ct_indices],
                                                                                    axis=-1),
                                                                             window_shape=(scale,scale),stride=(scale,scale), method='sum')

        test_mask = main_data['WSI']['reserve_index_image'][scale] & main_data['WSI']['out_roi_mask'][scale]
        test_data['test_mask'][scale] = test_mask
        test_data['fdr_mask'][scale] = test_mask & ct_data['wsi_ct_mask'][scale]
        test_data['test_yhat'][scale]=(test_data['total_pred_image'][scale][test_mask])
        test_data['test_y'][scale]=(test_data['total_true_image'][scale][test_mask])
        test_data['he_clusters'][scale]=(main_data['WSI']['he_clusters_image'][scale][test_mask])

    return test_data 


def load_inference_config(infer_json, query_type="gene_groups"):
    """
    Load targets_to_infer.json from data_path.
    Expected format:
        {
            "gene_groups": [
                {
                    "null_number": 0.2,
                    "groups": [
                        { "genes": ["EPCAM"] },
                        { "genes": ["EPCAM", "CDH1"], "name": "EPCAM_CDH1_meta" }
                    ]
                },
                {
                    "null_number": 0.01,
                    "groups": [
                        { "genes": ["MS4A1"] }
                    ]
                }
            ],
            "ct_groups": [
                {
                    "null_number": 0.1,
                    "groups": [
                        { "cell_types": ["Tumor"] },
                        { "cell_types": ["Tumor", "Epithelial"], "name": "Tumor_Epithelial_meta" }
                    ]
                },
                {
                    "null_number": 0.05,
                    "groups": [
                        { "cell_types": ["B_cell"] }
                    ]
                }
            ]
        }
    Returns:
        If query_type == "gene_groups": list of dicts with keys "genes" (list[str]),
            "name" (str), and "null_number" (float)
        If query_type == "ct_groups": list of dicts with keys "cell_types" (list[str]),
            "name" (str), and "null_number" (float)

    Raises:
        FileNotFoundError: if targets_to_infer.json is missing
        ValueError: if a multi-gene or multi-cell-type group has no "name" key
    """
    config_path = infer_json
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"{infer_json} file not found. "
            "Create this file to specify gene groups and null numbers."
        )
    with open(config_path) as f:
        config = json.load(f)
    
    if query_type == "gene_groups":
        gene_groups = []
        for null_group in config["gene_groups"]:
            null_number = null_group["null_number"]
            for entry in null_group["groups"]:
                genes = entry["genes"]
                if len(genes) == 1:
                    name = entry.get("name", genes[0])
                else:
                    if "name" not in entry:
                        raise ValueError(
                            f"Multi-gene group {genes} must have a 'name' key in targets_to_infer.json."
                        )
                    name = entry["name"]
                gene_groups.append({"genes": genes, "name": name, "null_number": null_number})
        return gene_groups
    if query_type == "ct_groups":
        ct_groups = []
        for null_group in config["ct_groups"]:
            null_number = null_group["null_number"]
            for entry in null_group["groups"]:
                cts = entry["cell_types"]
                if len(cts) == 1:
                    name = entry.get("name", cts[0])
                else:
                    if "name" not in entry:
                        raise ValueError(
                            f"Multi-cell-type group {cts} must have a 'name' key in targets_to_infer.json."
                        )
                    name = entry["name"]
                ct_groups.append({"cell_types": cts, "name": name, "null_number": null_number})
        return ct_groups
