#!/usr/bin/env python3

import numpy as np

def pca(X, ndim):
    
    X_norm = X - np.mean(X, axis = 0)

    U, S, Vh = np.linalg.svd(X_norm);
    
    S_diag = np.diag(S);
    return np.matmul(U[:, :ndim], S_diag[:ndim, :ndim])
