import numpy as np
from numpy.lib.stride_tricks import as_strided

def sliding_window(arr, window_shape=(4, 4), stride=(1, 1), method='sum',top_n=None):
    rows, cols = arr.shape
    window_height, window_width = window_shape
    stride_rows, stride_cols = stride
    
    out_rows = (rows - window_height) // stride_rows + 1
    out_cols = (cols - window_width) // stride_cols + 1
    
    if out_rows <= 0 or out_cols <= 0:
        raise ValueError("Window size too large for input array with given stride")
    
    windows = as_strided(
        arr,
        shape=(out_rows, out_cols, window_height, window_width),
        strides=(arr.strides[0] * stride_rows, 
                 arr.strides[1] * stride_cols,
                 arr.strides[0],
                 arr.strides[1])
    )
    
    if method == 'sum':
        return np.sum(windows, axis=(2, 3))
    elif method == 'mean':
        return np.mean(windows, axis=(2, 3))
    elif method == 'top_n_mean':
        if top_n is None:
            raise ValueError("top_n must be specified when method='top_n_mean'")
        
        # Validate top_n
        window_size = window_height * window_width
        if top_n > window_size:
            raise ValueError(f"top_n ({top_n}) cannot be larger than window size ({window_size})")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        
        # Reshape windows to flatten the window dimensions
        flat_windows = windows.reshape(out_rows, out_cols, -1)
        
        # Sort each window and take the top N values
        sorted_windows = np.sort(flat_windows, axis=2)
        top_n_values = sorted_windows[:, :, -top_n:]  # Take the last N (highest) values
        
        # Return the mean of the top N values
        return np.mean(top_n_values, axis=2)
    else:
        raise Exception("Unsupported method. Use 'sum', 'mean', or 'top_n_mean'.")

def sliding_window_3d(arr, window_shape=(4, 4), stride=(1, 1), method='sum'):

    height, width, channels = arr.shape
    window_height, window_width = window_shape
    stride_rows, stride_cols = stride
    
    out_height = (height - window_height) // stride_rows + 1
    out_width = (width - window_width) // stride_cols + 1
    
    if out_height <= 0 or out_width <= 0:
        raise ValueError("Window size too large for input array with given stride")
    
    windows = as_strided(
        arr,
        shape=(out_height, out_width, window_height, window_width, channels),
        strides=(arr.strides[0] * stride_rows, 
                 arr.strides[1] * stride_cols,
                 arr.strides[0],
                 arr.strides[1],
                 arr.strides[2])
    )
    
    if method == 'sum': # channel-wise summation
        return np.sum(windows, axis=(2, 3))
    if method == 'mean': # channel-wise mean
        return np.mean(windows, axis=(2, 3))
    raise Exception("Unsupported method.")



def reverse_sliding_window_boolean(arr_slide, output_shape, window_shape=(4, 4), stride=(1, 1)):
    out_rows, out_cols = arr_slide.shape
    window_height, window_width = window_shape
    stride_rows, stride_cols = stride
    
    orig_rows = (out_rows - 1) * stride_rows + window_height
    orig_cols = (out_cols - 1) * stride_cols + window_width
    
    reconstructed = np.zeros(output_shape, dtype=bool)
    
    for i in range(out_rows):
        for j in range(out_cols):
            if arr_slide[i, j]:
                start_row = i * stride_rows
                start_col = j * stride_cols
                
                end_row = start_row + window_height
                end_col = start_col + window_width
                reconstructed[start_row:end_row, start_col:end_col] = True
                
    return reconstructed