import argparse
import sys
from .read_data import load_main_data, load_gene_data, load_ct_data, load_inference_config
from .confidence import get_confidence_scores_for_all_cts
from .confidence import get_confidence_scores_for_all_genes


def run_for_gene(data_path,
                 infer_json,
                 main_data,
                 roi_calib_indices,
                 scales,
                 num_he_clusters,
                 visualize=True,
                 diagnostic=False):
    gene_data = load_gene_data(data_path=data_path, 
                               roi_calib_indices=roi_calib_indices, 
                               main_data=main_data)
    gene_groups = load_inference_config(infer_json=infer_json, query_type="gene_groups")
    get_confidence_scores_for_all_genes(main_data=main_data,
                                        gene_data=gene_data,
                                        gene_groups=gene_groups,
                                        data_path=data_path,
                                        roi_calib_indices=roi_calib_indices,
                                        scales=scales,
                                        num_he_clusters=num_he_clusters,
                                        visualize=visualize,
                                        diagnostic=diagnostic)
    return None


def run_for_ct(data_path,
               infer_json,
               main_data,
               roi_calib_indices,
               scales,
               num_he_clusters,
               visualize=True,
               diagnostic=False):
    ct_data = load_ct_data(data_path=data_path, 
                               roi_calib_indices=roi_calib_indices, 
                               main_data=main_data,
                               scales=scales)
    ct_groups = load_inference_config(infer_json=infer_json, query_type="ct_groups")
    get_confidence_scores_for_all_cts(main_data=main_data,
                                        ct_data=ct_data,
                                        ct_groups=ct_groups,
                                        data_path=data_path,
                                        roi_calib_indices=roi_calib_indices,
                                        scales=scales,
                                        num_he_clusters=num_he_clusters,
                                        visualize=visualize,
                                        diagnostic=diagnostic)
    return None
    
def run(data_path,
        infer_json,
        num_he_clusters,
        roi_calib_indices,
        scales,
        visualize=True,
        diagnostic=False):
    
    if isinstance(roi_calib_indices, int):
        roi_calib_indices = [roi_calib_indices]

    main_data = load_main_data(data_path=data_path, num_he_clusters=num_he_clusters, roi_calib_indices=roi_calib_indices, scales=scales)

    print("Running inference for gene queries...")
    run_for_gene(data_path=data_path,
                 infer_json=infer_json,
                 main_data=main_data,
                 roi_calib_indices=roi_calib_indices,
                 num_he_clusters=num_he_clusters,
                 scales=scales,
                 visualize=visualize,
                 diagnostic=diagnostic)
    
    print("Running inference for cell type queries...")
    run_for_ct(data_path=data_path,
               infer_json=infer_json,
               main_data=main_data,
               roi_calib_indices=roi_calib_indices,
               num_he_clusters=num_he_clusters,
               scales=scales,
               visualize=visualize,
               diagnostic=diagnostic)
    

def main():
    parser = argparse.ArgumentParser(description="Run UTOPIA inference pipeline.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--infer_json", type=str, required=True,
                        help="Inference JSON file path.")
    parser.add_argument("--num_he_clusters", type=int, required=True,
                        help="Number of H&E clusters.")
    parser.add_argument('--roi_calib_indices', type=int, nargs='+', required=True,
                        help='ROI indices for calibration (single integer or space-separated list, e.g., 0 or 0 1 2)')
    parser.add_argument("--scales", type=int, nargs="+", required=True,
                        help="Scale values (space-separated list of floats).")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Enable visualization (default: True).")
    parser.add_argument("--diagnostic", action="store_true", default=False,
                        help="Enable diagnostic mode (default: False).")
    args = parser.parse_args()

    run(data_path=args.data_path,
        infer_json=args.infer_json,
        num_he_clusters=args.num_he_clusters,
        roi_calib_indices=args.roi_calib_indices,
        scales=args.scales,
        visualize=args.visualize,
        diagnostic=args.diagnostic)


if __name__ == "__main__":
    sys.exit(main())
