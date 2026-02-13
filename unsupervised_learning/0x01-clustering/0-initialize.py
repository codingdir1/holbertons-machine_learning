#!/usr/bin/env python3

import numpy as np

def initialize(X, k):
    if k <= 0:
        return None

    return np.random.uniform(low = np.min(X, axis = 0), 
        high = np.max(X, axis = 0), 
        size = (k, X.shape[1]))
