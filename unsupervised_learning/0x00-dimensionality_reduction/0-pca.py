#!/usr/bin/env python3

import numpy as np

def pca(X, var = 0.95):
    
    U, S, Vh = np.linalg.svd(X);
    V = Vh.T

    cumulative = np.cumsum(S) / np.sum(S);
    i = (cumulative >= var).argmax()

    W = V[:, :(i + 1)]
    return W
