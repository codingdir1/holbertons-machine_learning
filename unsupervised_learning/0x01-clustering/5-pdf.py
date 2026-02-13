#!/usr/bin/env python3

import numpy as np

def pdf(X, m, S):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(m, np.ndarray) or len(m.shape) != 1 or \
        not isinstance(S, np.ndarray) or len(S.shape) != 2 or \
        not(X.shape[1] == m.shape[0] == S.shape[0] == S.shape[1]):
        return None
    
    d = X.shape[1]

    inv_S = np.linalg.inv(S)
    normalizer = 1.0 / (((2 * np.pi) ** (d / 2)) * (np.linalg.det(S) ** 0.5))
    exponent = -0.5 * np.sum(np.matmul(X - m, np.linalg.inv(S)) * (X - m), axis = 1)
    P = normalizer * np.exp(exponent)
    return P + 1e-300
