#!/usr/bin/py python3

import numpy as np

def kmeans(X, k, iterations = 1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(k, int) or k <= 0 or k > X.shape[0] or \
        not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = np.random.uniform(low = np.min(X, axis = 0), 
            high = np.max(X, axis = 0), 
            size = (k, X.shape[1]))
    
    for i in range(iterations):
        distances  = np.linalg.norm(X[:, None, :] - C, axis = 2)
        clss = np.argmin(distances, axis = 1)

        C_new = np.empty((k, X.shape[1]))
        for j in range(k):
            cluster = X[clss == j]
            if cluster.size == 0:
                C_new[j] = np.random.uniform(low = np.min(X, axis = 0),
                    high = np.max(X, axis = 0),
                    size = (1, X.shape[1]))
            else:
                C_new[j] = np.mean(cluster, axis = 0)

        distances = np.linalg.norm(X[:, None, :] - C_new, axis = 2)
        clss = np.argmin(distances, axis = 1)

        if np.array_equal(C_new, C):
            return C, clss

        C = C_new
        
    return None, None
