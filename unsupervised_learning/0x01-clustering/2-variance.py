#!/usr/bin/env pythom3

import numpy as np

def variance(X, C):
    if not isinstance(X, np.ndarray) or \
        not isinstance(C, np.ndarray) or \
        len(X.shape) != 2 or len(C.shape) != 2 or \
        X.shape[0] < C.shape[0] or \
        X.shape[1] != C.shape[1]:
        return None
    
    min_distances = np.min(np.linalg.norm(X[:, None, :] - C, axis = 2), 
        axis = 1)
    return np.sum(np.square(min_distances))
