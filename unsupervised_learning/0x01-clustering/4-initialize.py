#!/usr/bin/env python3

import numpy as np

kmeans = __import__('1-kmeans').kmeans

def initialize(X, k):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(k, int) or k <= 0:
        return None, None, None

    pi = np.full(k, 1 / k)
    m, clss = kmeans(X, k)
    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))
    
    return pi, m, S
