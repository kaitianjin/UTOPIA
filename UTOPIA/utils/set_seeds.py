
import numpy as np
import torch
import random

def set_seeds(seed):
    """
    Set seeds for all sources of randomness in Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use
    """
    
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Additional PyTorch settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False