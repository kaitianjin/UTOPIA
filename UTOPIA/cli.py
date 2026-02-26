import argparse
import configparser
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="UTOPIA spatial pathology pipeline")
    parser.add_argument('--config', type=str, 
                        help='Path to utopia_config file')
    parser.add_argument('--rescale_raw_he',      action='store_true', help='Rescale raw H&E image')
    parser.add_argument('--he_raw', type=str,
                        help='Path to the original raw H&E image (e.g. "data/raw/he.tiff"); required with --rescale_raw_he')
    parser.add_argument('--pixel_size_raw', type=str,
                        help='Pixel size of the original raw H&E image in microns (e.g. 0.25); required with --rescale_raw_he')
    parser.add_argument('--get_tissue_mask',     action='store_true', help='Get tissue mask')
    parser.add_argument('--process_roi_data',    action='store_true', help='Prepare transcript/cell-type data for ROI')
    parser.add_argument('--feature_extraction',  action='store_true', help='UNI feature extraction')
    parser.add_argument('--histology_clustering',action='store_true', help='Histology clustering')
    parser.add_argument('--calibration',         action='store_true', help='Calibration')
    parser.add_argument('--inference',           action='store_true', help='Inference')
    args = parser.parse_args()

    if args.rescale_raw_he and (not args.he_raw or args.pixel_size_raw is None):
        parser.error('--he_raw and --pixel_size_raw are required when --rescale_raw_he is set')

    def run(cmd):
        print(f"[UTOPIA] Running: {' '.join(str(c) for c in cmd)}")
        subprocess.run(cmd, check=True)
    py = sys.executable

    # ------------------------------------------------------------------
    # Rescale original raw H&E image to 0.5 um/pixel
    # ------------------------------------------------------------------
    if args.rescale_raw_he:
        run([py, '-m', 'UTOPIA.img_process.scaling_he_image',
             '--he_raw', args.he_raw,
             '--pixel_size_raw', args.pixel_size_raw])
        return
        
    steps = {
        'rescale_raw_he':       args.rescale_raw_he,
        'get_tissue_mask':      args.get_tissue_mask,
        'process_roi_data':     args.process_roi_data,
        'feature_extraction':   args.feature_extraction,
        'histology_clustering': args.histology_clustering,
        'calibration':          args.calibration,
        'inference':            args.inference,
    }
    run_all = not any(steps.values())

    cfg = configparser.ConfigParser()
    if args.config is None:
        parser.error('--config is required')
    cfg.read(args.config)

    # Global
    data_dir  = cfg['global']['data_dir']
    model_dir = cfg['global']['model_dir']

    # histosweep
    min_size  = cfg['histosweep'].get('min_size', '50')

    # ROI  (separators kept as-is: "row,col;row,col" and "w1,w2")
    num_roi = cfg['roi']['num_roi']
    roi_top_left = cfg['roi']['top_left']
    roi_width    = cfg['roi']['width']
    roi_height   = cfg['roi']['height']

    # Scale
    scales         = cfg['scale']['scales']           # e.g. "1 2 4"
    emb_scales = ' '.join(s for s in scales.split() if s != '1')  # e.g. "2 4"

    # Clustering
    n_clusters = cfg['clustering']['n_clusters']

    # Calibration  (indices are space-separated strings: "0 1")
    roi_train_indices = cfg['calibration']['roi_train_indices']
    roi_calib_indices = cfg['calibration']['roi_calib_indices']
    parallel_procs    = cfg['calibration'].get('parallel_processes', '8')

    # Predict flags
    predict_gene      = cfg['predict'].getboolean('gene', True)
    predict_cell_type = cfg['predict'].getboolean('cell_type', True)
    if predict_cell_type:
        n_cell_types = cfg['calibration']['n_cell_types']

    infer_json = cfg['inference']['infer_json']

    roi_args = [
        '--roi-top-left', roi_top_left,
        '--roi-width', roi_width,
        '--roi-height', roi_height,
    ]

    # ------------------------------------------------------------------
    # 1. Get tissue mask
    # ------------------------------------------------------------------
    if run_all or steps['get_tissue_mask']:
        run([py, '-m', 'UTOPIA.img_process.get_tissue_mask',
             '--data_path', data_dir,
             '--clean_background',
             '--min_size', min_size])

        run([py, '-m', 'UTOPIA.utils.crop_roi',
             '--data_path', data_dir, '--he_mask', *roi_args])

        run([py, '-m', 'UTOPIA.img_process.scaling_image_features',
             '--data_path', data_dir,
             '--scale', *scales.split(),
             '--he_mask', '--wsi', '--roi', '--num_roi', num_roi])
        
    # ------------------------------------------------------------------
    # 2. UNI feature extraction
    # ------------------------------------------------------------------
    if run_all or steps['feature_extraction']:
        run([py, '-m', 'UTOPIA.img_process.feature_extraction_uni',
             '--data_path', data_dir,
             '--model_path', model_dir])

        run([py, '-m', 'UTOPIA.utils.crop_roi',
             '--data_path', data_dir, '--emb', *roi_args])

        run([py, '-m', 'UTOPIA.img_process.scaling_image_features',
             '--data_path', data_dir,
             '--scale', *emb_scales.split(),
             '--emb', '--wsi', '--roi', '--num_roi', num_roi])

        run([py, '-m', 'UTOPIA.img_process.reduce_dimensions',
             '--data_paths', data_dir,
             '--scale', *scales.split(),
             '--wsi', '--roi', '--clean', '--num_roi', num_roi])
     
    # ------------------------------------------------------------------
    # 3. Prepare transcript / cell-type data for ROI
    # ------------------------------------------------------------------
    if run_all or steps['process_roi_data']:
        if predict_gene:
            run([py, '-m', 'UTOPIA.utils.crop_roi',
                 '--data_path', data_dir, '--tr_image', *roi_args])

        if predict_cell_type:
            # --ct_image and --ct_mask can be passed in one call
            run([py, '-m', 'UTOPIA.utils.crop_roi',
                 '--data_path', data_dir,
                 '--ct_image', '--ct_mask', *roi_args])

            run([py, '-m', 'UTOPIA.img_process.scaling_image_features',
                 '--data_path', data_dir,
                 '--scale', *scales.split(),
                 '--ct_mask', '--wsi', '--roi', '--num_roi', num_roi])
          


    # ------------------------------------------------------------------
    # 4. Histology clustering
    # ------------------------------------------------------------------
    if run_all or steps['histology_clustering']:
        run([py, '-m', 'UTOPIA.img_process.histology_clustering',
             '--data_paths', data_dir,
             '--scale', *scales.split(),
             '--wsi', '--roi', 
             '--n_clusters', n_clusters])

    # ------------------------------------------------------------------
    # 5. Calibration
    # ------------------------------------------------------------------
    if run_all or steps['calibration']:
        run([py, '-m', 'UTOPIA.calibration.generate_folds',
             data_dir,                      # positional arg in generate_folds
             '--n_clusters', n_clusters])

        if predict_cell_type:
            run([py, '-m', 'UTOPIA.calibration.predict_ct',
                 '--data_path', data_dir,
                 '--roi_train_indices', *roi_train_indices.split(),
                 '--roi_calib_indices', *roi_calib_indices.split(),
                 '--n_cell_types', n_cell_types,
                 '--parallel_processes', parallel_procs])

        if predict_gene:
            run([py, '-m', 'UTOPIA.calibration.predict_gene',
                 '--data_path', data_dir,
                 '--roi_train_indices', *roi_train_indices.split(),
                 '--roi_calib_indices', *roi_calib_indices.split(),
                 '--parallel_processes', parallel_procs])

    # ------------------------------------------------------------------
    # 6. Inference
    # ------------------------------------------------------------------
    if run_all or steps['inference']:
        run([py, '-m', 'UTOPIA.inference.run',
             '--data_path', data_dir,
             '--infer_json', infer_json,
             '--num_he_clusters', n_clusters,
             '--roi_calib_indices', *roi_calib_indices.split(),
             '--scales', *scales.split(),
             '--visualize'])


if __name__ == '__main__':
    main()
