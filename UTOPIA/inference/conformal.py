import numpy as np

def bh_adjust(pvals):
    m = len(pvals)
    if m == 0:
        return np.array([])
    
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    
    adjusted_pvals = m * sorted_pvals / np.arange(1, m+1)
    
    # Enforce monotonicity
    for i in range(m-2, -1, -1):
        adjusted_pvals[i] = min(adjusted_pvals[i], adjusted_pvals[i+1])
    
    adjusted_pvals = np.clip(adjusted_pvals, 0, 1)
    
    result = np.empty_like(adjusted_pvals)
    result[sorted_indices] = adjusted_pvals
    return result

def compute_pvals(test_conformity, calib_conformity, tolerance=1e-2):
    # Time complexity: O(n log n + m log n) where n = len(calib_conformity), m = len(test_conformity)
    sorted_calib = np.sort(calib_conformity + tolerance)
    n = len(calib_conformity)
    num_smaller = np.searchsorted(sorted_calib, test_conformity, side='left')
    pvals = (num_smaller + 1) / (n + 1)
    return pvals


# def nonconformity_score_res(y, yhat, c=0):
#     return y - yhat 

def nonconformity_score_clip(y, yhat, c=0):
    scores = np.ones(len(y))
    scores[y<=c] = (-yhat)[y<=c]
    scores[y>c] = np.inf 
    return scores

# def nonconformity_score_clip_res(y, yhat, c=0):
#     scores = np.ones(len(y))
#     scores[y<=c] = (y-yhat)[y<=c]
#     scores[y>c] = np.inf 
#     return scores

# def nonconformity_score_quotient(y, yhat, c=0): 
#     scores = np.ones(len(y)) 
#     scores[y<=c] = ((y+1)/yhat)[y<=c]
#     scores[y>c] = np.inf 
#     return scores

def conformal_pvals(calib_y, 
                    calib_yhat, 
                    test_yhat, 
                    null, 
                    nonconformity_score=nonconformity_score_clip, 
                    tolerance = 0):

    calib_scores = nonconformity_score(calib_y, calib_yhat, null)
    test_scores = nonconformity_score(null * np.ones(len(test_yhat)), test_yhat, null)
    pvals = compute_pvals(test_scores, calib_scores, tolerance=tolerance)
    return bh_adjust(pvals)



def compute_tolerance(calib_yhat, calib_y, test_yhat, null_number, correct_batch_effect_magnitude): 
    if isinstance(correct_batch_effect_magnitude, float):
        tolerance = correct_batch_effect_magnitude
    if isinstance(correct_batch_effect_magnitude, int):
        tolerance = correct_batch_effect_magnitude

    quantile = np.sum(calib_y<=null_number)/len(calib_y)
    if quantile > 0.1:
        calib_sorted = np.sort(calib_yhat)
        test_sorted = np.sort(test_yhat)
        calib_q_threshold = np.quantile(calib_sorted, quantile)
        test_q_threshold = np.quantile(test_sorted, quantile)
        calib_first_q = calib_sorted[calib_sorted <= calib_q_threshold]
        test_first_q = test_sorted[test_sorted <= test_q_threshold]

        if correct_batch_effect_magnitude == "weak": 
            quantiles = np.array([0.01,0.1,0.3,0.5,0.7,0.9,0.99])
        elif correct_batch_effect_magnitude == "medium": 
            quantiles = np.array([0.001,0.005,0.01,0.1,0.3,0.5,0.7,0.9,0.99,0.995,0.999]) 
        elif correct_batch_effect_magnitude == "strong": 
            quantiles = np.array([0.001,0.005,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.995,0.999]) 
        else: 
            raise ValueError("correct_batch_effect_magnitude must be float, int, or strings in ['weak','medium','strong']") 
            
        calib_quantiles = np.quantile(calib_first_q, quantiles)
        test_quantiles = np.quantile(test_first_q, quantiles)
        quantile_shifts = np.clip(test_quantiles - calib_quantiles,a_min=0,a_max=None)
        tolerance = -np.max(quantile_shifts)
    else:
        tolerance = 0 
    return tolerance 


def get_adj_pvals(calib_yhats,
           calib_ys,
           test_yhats, 
           calib_he_cls,
           test_he_cls,
           null_number,
           calib_he_clusters,
           min_calib_pts=None,
           correct_batch_effect=True,
           correct_batch_effect_magnitude="medium",
           verbose=False):
    
    adjust_pvals = np.ones_like(test_yhats)
    
    for cluster in calib_he_clusters:
        
        calib_yhat = calib_yhats[calib_he_cls==cluster] 
        calib_y = calib_ys[calib_he_cls==cluster] 
        test_yhat = test_yhats[test_he_cls==cluster] 

        if min_calib_pts is None: 
            min_calib_pts = max(200, len(test_yhat)*0.01)
        if verbose:
            print(f"Minimum calibration points for cluster {cluster} is {min_calib_pts}.")
            if len(calib_yhat) < min_calib_pts: 
                print(f"Skip cluster {cluster} because the calibration data only contains {len(calib_yhat)} points.")
                continue
                
        if not correct_batch_effect: 
            tolerance = 0 
        if correct_batch_effect and correct_batch_effect_magnitude: 
            tolerance = compute_tolerance(calib_yhat, calib_y, test_yhat, null_number, correct_batch_effect_magnitude)
        
        adjust_pvals[test_he_cls==cluster] = conformal_pvals(calib_y,
                                                             calib_yhat, 
                                                             test_yhat,
                                                             null_number,
                                                             tolerance=tolerance)
    return adjust_pvals


    
