#!/usr/bin/env python3

import numpy as np

def P_init(X, perplexity):
    
    diff = X[:, None, :] - X[None, :, :]
    D = np.sum(diff ** 2, axis = -1)
    
    n = X.shape[0]
    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return (D, P, betas, H)
