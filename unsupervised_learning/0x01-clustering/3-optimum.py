#!/usr/bin/env python3

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance

def optimum_k(X, kmin = 1, kmax = None, iterations = 1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(kmin, int) or kmin <= 0 or \
        not isinstance(kmax, int) or kmax <= kmin or \
        not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []

    if kmax == None:
        kmax = 2

    for i in range(kmin, kmax + 1, 1):
        C, clss = kmeans(X, i, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, C))
    
    smallest = d_vars[0]
    for i in range(len(d_vars)):
        d_vars[i] = smallest - d_vars[i]

    return results, d_vars
