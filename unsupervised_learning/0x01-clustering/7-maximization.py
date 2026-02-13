#!/usr/bin/env python3

import numpy as np

def maximization(X, g):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(g, np.ndarray) or len(g.shape) != 2 or \
        X.shape[0] != g.shape[1]:
        return None, None, None

    g_sum = np.sum(g, axis = 1)

    pi = g_sum / X.shape[0]

    m = np.matmul(g, X) / g_sum[:, None]

    diff = X[None, :, :] - m[:, None, :]
    S = (diff.transpose(0, 2, 1) @ (g[:, :, None] * diff)) / g_sum[:, None, None]
    
    return pi, m, S
