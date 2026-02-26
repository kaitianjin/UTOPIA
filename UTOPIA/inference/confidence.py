import os

from .plot import convert_binary_to_rgb, convert_confidence_to_rgb, convert_magnitude_to_rgb, plot_rgb_image
from .read_data import load_gene_calib_data, load_gene_test_data
from .read_data import load_ct_calib_data, load_ct_test_data
from .conformal import get_adj_pvals
from .fdr import get_fdp_control_curves, save_fdr_data_to_csv
from collections import defaultdict
import numpy as np

def convert_pvals_to_confidence_scores(pvals, p_min=0.01, p_max=0.5, log_base=10):
    log = lambda x: np.log(x) / np.log(log_base)
    scores = (-log(pvals) + log(p_max)) / (-log(p_min) + log(p_max))
    return np.clip(scores, 0, 1)

def get_confidence_scores(calib_data,
                          test_data,
                          scales,
                          num_he_clusters,
                          null_number,
                          result_path=None,
                          diagnostic=False,
                          main_data=None):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for scale in scales: 
        results["pvals_adjust_image"][scale] = np.ones(test_data["total_pred_image"][scale].shape).astype(float)
        
        adjust_pvals = get_adj_pvals(
            calib_yhats=calib_data['calib_yhat'][scale],
            calib_ys=calib_data['calib_y'][scale], 
            test_yhats=test_data['test_yhat'][scale], 
            calib_he_cls=calib_data['he_clusters'][scale], 
            test_he_cls=test_data['he_clusters'][scale], 
            null_number=null_number, 
            calib_he_clusters = range(num_he_clusters)) 
        
        # 2d numpy array of adjusted p values
        test_mask = test_data['test_mask'][scale]
        results["pvals_adjust_image"][scale][test_mask] = adjust_pvals 
        if result_path is not None:
            os.makedirs(os.path.join(result_path, "raw_files"), exist_ok=True)
            np.save(os.path.join(result_path, 'raw_files', f"pvals_adj_image_null_{null_number}_scale_{scale}.npy"), results["pvals_adjust_image"][scale])

        # 2d numpy array of confidence scores
        confidence_scores = convert_pvals_to_confidence_scores(adjust_pvals) 
        results["confidence_score_image"][scale] = np.ones(test_data["total_pred_image"][scale].shape).astype(float)
        results["confidence_score_image"][scale][test_mask] = confidence_scores
        if result_path is not None:
            os.makedirs(os.path.join(result_path, "raw_files"), exist_ok=True)
            np.save(os.path.join(result_path, 'raw_files', f"confidence_score_image_null_{null_number}_scale_{scale}.npy"), results["confidence_score_image"][scale])

        if diagnostic and main_data is not None and result_path is not None:
            fdr_values, fdp_values, n_discover = get_fdp_control_curves(results["pvals_adjust_image"][scale],
                                                                test_data['total_true_image'][scale], 
                                                                test_data['fdr_mask'][scale],
                                                                null_number)
            results["fdr_values"][scale] = fdr_values
            results["fdp_values"][scale] = fdp_values
            results["n_discover"][scale] = n_discover

    return results 



def get_confidence_scores_for_gene(main_data,
                                   gene_data,
                                   genes,
                                   roi_calib_indices,
                                   num_he_clusters,
                                   scales,
                                   null_number,
                                   result_path=None,
                                   visualize=True,
                                   diagnostic=False):
    # get confidence scores for a single gene or a meta-gene 
    # and visualize the confidence score images

    calib_data = load_gene_calib_data(main_data, 
                                      gene_data, 
                                      genes, 
                                      roi_calib_indices,
                                      scales)
    test_data = load_gene_test_data(main_data,
                                    gene_data,  
                                    calib_data,
                                    genes,
                                    scales)
    confidence_results = get_confidence_scores(calib_data,
                                               test_data,
                                               scales,
                                               num_he_clusters=num_he_clusters,
                                               null_number=null_number,
                                               result_path=result_path,
                                               diagnostic=diagnostic,
                                               main_data=main_data)
                                               
    if visualize and result_path is not None:
        for scale in scales:
            conf_rgb = convert_confidence_to_rgb(confidence_results["confidence_score_image"][scale],
                                                main_data['WSI']['reserve_index_image'][scale],
                                                test_data['total_true_image'][scale],
                                                main_data['WSI']['roi_mask'][scale],
                                                true_type='magnitude')
            plot_rgb_image(conf_rgb, 
                        save_path=os.path.join(result_path,
                                                f"confidence_score_image_null_{null_number}_scale_{scale}.png"))
            pred_rgb = convert_magnitude_to_rgb(test_data['total_pred_image'][scale],
                                                main_data['WSI']['reserve_index_image'][scale],
                                                test_data['total_true_image'][scale],
                                                main_data['WSI']['roi_mask'][scale],
                                                true_type='magnitude')
            plot_rgb_image(pred_rgb, 
                        save_path=os.path.join(result_path, 
                                                f"pred_image_scale_{scale}.png"))
        
            if diagnostic:
                os.makedirs(os.path.join(result_path, "diagnostic"), exist_ok=True)
                true_rgb = convert_magnitude_to_rgb(test_data['total_true_image'][scale],
                                                main_data['WSI']['reserve_index_image'][scale],
                                                test_data['total_true_image'][scale],
                                                main_data['WSI']['roi_mask'][scale],
                                                true_type='magnitude')
                plot_rgb_image(true_rgb, 
                            save_path=os.path.join(result_path, "diagnostic",
                                                    f"true_image_scale_{scale}.png"))
    
    # output fdr control results
    if diagnostic and result_path is not None:
        os.makedirs(os.path.join(result_path, "diagnostic"), exist_ok=True)
        save_fdr_data_to_csv(confidence_results["fdp_values"],
                             confidence_results["fdr_values"],
                             confidence_results["n_discover"],
                             output_file=os.path.join(result_path, "diagnostic",
                                                      f"fdr_control_data_null_{null_number}.csv"),
                             scales=scales)
    
    return None
    
def get_confidence_scores_for_ct(main_data,
                                ct_data,
                                ct_groups,
                                roi_calib_indices,
                                num_he_clusters,
                                scales,
                                null_number,
                                result_path=None,
                                visualize=True,
                                diagnostic=False):
    # get confidence scores for a single ct group
    # and visualize the confidence score images


    calib_data = load_ct_calib_data(main_data, 
                                      ct_data, 
                                      ct_groups, 
                                      roi_calib_indices,
                                      scales)
    test_data = load_ct_test_data(main_data,
                                    ct_data,  
                                    ct_groups,
                                    scales)

    confidence_results = get_confidence_scores(calib_data,
                                               test_data,
                                               scales,
                                               num_he_clusters=num_he_clusters,
                                               null_number=null_number,
                                               result_path=result_path,
                                               diagnostic=diagnostic,
                                               main_data=main_data)
                                               
    if visualize and result_path is not None:
        for scale in scales:
            conf_rgb = convert_confidence_to_rgb(confidence_results["confidence_score_image"][scale],
                                                main_data['WSI']['reserve_index_image'][scale],
                                                test_data['total_true_image'][scale]>null_number,
                                                main_data['WSI']['roi_mask'][scale],
                                                true_type='cell type',
                                                positive_color=[255, 0, 0])
            plot_rgb_image(conf_rgb, 
                        save_path=os.path.join(result_path,
                                                f"confidence_score_image_null_{null_number}_scale_{scale}.png"))
            pred_rgb = convert_magnitude_to_rgb(test_data['total_pred_image'][scale],
                                                main_data['WSI']['reserve_index_image'][scale],
                                                test_data['total_true_image'][scale]>null_number,
                                                main_data['WSI']['roi_mask'][scale],
                                                true_type='cell type',
                                                positive_color=[255, 0, 0])
            plot_rgb_image(pred_rgb, 
                        save_path=os.path.join(result_path, 
                                                f"softmax_image_null_{null_number}_scale_{scale}.png"))
        
            if diagnostic:
                os.makedirs(os.path.join(result_path, "diagnostic"), exist_ok=True)
                true_rgb = convert_binary_to_rgb(test_data['total_true_image'][scale]>null_number,
                                                main_data['WSI']['reserve_index_image'][scale],
                                                positive_color=[255, 0, 0])
                plot_rgb_image(true_rgb, 
                            save_path=os.path.join(result_path, "diagnostic",
                                                    f"true_image_null_{null_number}_scale_{scale}.png"))
    
    # output fdr control results
    if diagnostic and result_path is not None:
        os.makedirs(os.path.join(result_path, "diagnostic"), exist_ok=True)
        save_fdr_data_to_csv(confidence_results["fdp_values"],
                             confidence_results["fdr_values"],
                             confidence_results["n_discover"],
                             output_file=os.path.join(result_path, "diagnostic",
                                                      f"fdr_control_data_null_{null_number}.csv"),
                             scales=scales)
    
    return None


def get_confidence_scores_for_all_genes(main_data,
                                        gene_data,
                                        gene_groups,
                                        data_path,
                                        num_he_clusters,
                                        roi_calib_indices,
                                        scales,
                                        visualize=True,
                                        diagnostic=False):
    
    os.makedirs(os.path.join(data_path, "results"), exist_ok=True)

    for group in gene_groups:
        genes = group["genes"]
        name = group["name"]
        null_number = group["null_number"]
        print(f"Processing gene/meta-gene {name} ({genes}), null_number={null_number}...")

        result_path = os.path.join(data_path, "results", name)
        os.makedirs(result_path, exist_ok=True)
        get_confidence_scores_for_gene(main_data=main_data,
                                       gene_data=gene_data,
                                       genes=genes,
                                       num_he_clusters=num_he_clusters,
                                       roi_calib_indices=roi_calib_indices,
                                       scales=scales,
                                       null_number=null_number,
                                       result_path=result_path,
                                       visualize=visualize,
                                       diagnostic=diagnostic)
    
    return None

def get_confidence_scores_for_all_cts(main_data,
                                        ct_data,
                                        ct_groups,
                                        data_path,
                                        num_he_clusters,
                                        roi_calib_indices,
                                        scales,
                                        visualize=True,
                                        diagnostic=False):
    
    os.makedirs(os.path.join(data_path, "results"), exist_ok=True)

    for group in ct_groups:
        ct_groups = group["cell_types"]
        name = group["name"]
        null_number = group["null_number"]
        print(f"Processing cell type/composite cell type {name} ({ct_groups}), null_number={null_number}...")

        result_path = os.path.join(data_path, "results", name)
        os.makedirs(result_path, exist_ok=True)
        get_confidence_scores_for_ct(main_data=main_data,
                                     ct_data=ct_data,
                                       ct_groups=ct_groups,
                                       num_he_clusters=num_he_clusters,
                                       roi_calib_indices=roi_calib_indices,
                                       scales=scales,
                                       null_number=null_number,
                                       result_path=result_path,
                                       visualize=visualize,
                                       diagnostic=diagnostic)
    
    return None