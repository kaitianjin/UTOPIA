import numpy as np 
import pandas as pd
from typing import Dict, List
def get_fdp_single_fdr(pval_image, true_image, mask, null, fdr):
    # return the false discovery proportion and the number of discoveries
    pval_pos = pval_image[mask] <= fdr # set of discoveries
    true_pos = true_image[mask] > null  # set of true positives
    n_pos = np.sum(pval_pos)
    fdp = 0.0 if n_pos == 0 else 1 - np.sum(true_pos & pval_pos) / n_pos
    return fdp, n_pos

def get_fdp_control_curves(pvals_adjust_image, 
                           total_true_gene_image, 
                           mask, 
                           null, 
                           plot = False,
                           fdr_values= np.arange(0.01, 1.00, 0.01)): 
    
    get_fdp_output = [get_fdp_single_fdr(pvals_adjust_image, 
                                         total_true_gene_image, 
                                         mask, 
                                         null, 
                                         fdr) for fdr in fdr_values]
    fdp_values = [output[0] for output in get_fdp_output]
    n_discover = [output[1] for output in get_fdp_output]

    if plot: 
        import matplotlib.pyplot as plt

        min_dscv = np.sum(mask) * 0.01  
        # only plot points with at least 1% discoveries to avoid 
        # plotting points with very few discoveries which can be unstable
        plot_fdr_values = fdr_values[np.array(n_discover) >= min_dscv]
        plot_fdp_values = np.array(fdp_values)[np.array(n_discover) >= min_dscv]
        
        fig, ax1 = plt.subplots(figsize=(5, 5))
        color1 = 'tab:blue'
        ax1.set_xlabel('Set FDR', fontsize=12)
        ax1.set_ylabel('Realized FDP', color=color1, fontsize=12)
        ax1.plot(plot_fdr_values, plot_fdp_values, color=color1, linewidth=2, label='Realized FDP')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        plt.title('Realized FDP and Power vs Set FDR', fontsize=14)
        fig.tight_layout()
        plt.show()

    return fdr_values, fdp_values, n_discover


def save_fdr_data_to_csv(fdp_dict: Dict[int, List[float]],
                         fdr_dict: Dict[int, List[float]],
                         n_discover_dict: Dict[int, List[int]],
                         output_file: str,
                         scales: List[int]):
    
    # Since fdr_values are the same across scales, use the first one
    fdr_values = fdr_dict[scales[0]]
    
    data = {'fdr': fdr_values}
    for scale in scales:
        data[f'fdp_{scale}'] = fdp_dict[scale]
        data[f'n_discover_{scale}'] = n_discover_dict[scale]
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    return df
