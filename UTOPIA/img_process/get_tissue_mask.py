from .HistoSweep.computeMetrics import compute_metrics
from .HistoSweep.densityFiltering import compute_low_density_mask
from .HistoSweep.textureAnalysis import run_texture_analysis
from .HistoSweep.ratioFiltering import run_ratio_filtering
from .HistoSweep.generateMask import generate_final_mask
from ..utils.io import load_image 
import argparse
import numpy as np 
import sys
import os

def generate_filter_mask( 
        img:np.ndarray,
        output_path: str = None,
        density_thresh: float = 100,
        min_size: int = 10,
        patch_size: int = 16,
        clean_background: bool = False
):
        he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics(img, patch_size=patch_size) 
        mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
        mask1_lowdensity_update = run_texture_analysis(prefix=output_path, image=img, tissue_mask=mask1_lowdensity,  glcm_levels=64)
        mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
        os.makedirs(os.path.join(output_path, "mask"), exist_ok=True)
        generate_final_mask(prefix=output_path, he=img, output_dir="mask",
                            mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio, 
                            clean_background = clean_background, 
                            super_pixel_size=patch_size, minSize = min_size)
        return None 


def get_args():
        parser = argparse.ArgumentParser(description="Generate filtering mask for H&E images using HistoSweep.")
        parser.add_argument("--data_path", type=str, required=True, help="Path to the folder containing the rescaled and padded H&E image.")
        parser.add_argument("--density_thresh", type=float, default=100, help="Density threshold for low-density mask.")
        parser.add_argument("--min_size", type=int, default=10, help="Minimum size of connected components to keep.")
        parser.add_argument("--patch_size", type=int, default=16, help="Patch size for texture analysis.")
        parser.add_argument("--clean_background", action='store_true', help="Flag to clean background in the final mask.")
        return parser.parse_args()

def main():
        args = get_args()
        filename = os.path.join(args.data_path, 'HE', 'he.png')
        img = load_image(filename, if_ome_tif=False, if_svs=False)
        img = img.astype(np.float32)

        generate_filter_mask(
                img=img,
                output_path=args.data_path,
                density_thresh=args.density_thresh,
                min_size=args.min_size,
                patch_size=args.patch_size,
                clean_background=args.clean_background
                )
        
if __name__ == "__main__":
        sys.exit(main())

