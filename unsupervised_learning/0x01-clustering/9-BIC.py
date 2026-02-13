#!/usr/bin/env python3

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin = 1, kmax = None, iterations = 1000, tol = 1e-5, verbose = False):
    n, d = X.shape
    if kmax == None:
        k_max = n

    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(kmin, int) or kmin <= 0 or \
        not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin or \
        not isinstance(tol, float) or tol < 0.0 or \
        not isinstance(verbose, bool):
        return None, None, None, None
    
    best_k, best_result, l, b = None, None, [], []

    k = kmin
    BIC_prev = None
    while k <= kmax:
        pi, m, S, g, l_k = expectation_maximization(X, k, iterations, tol, verbose)
        p = k * (d + (d * (d + 1) / 2)) + k - 1
        l.append(l_k)
        
        BIC = p * np.log(n) - 2 * l_k
        b.append(BIC)

        if k == kmin:
            BIC_prev = BIC
        elif BIC < BIC_prev:
            best_k = k
            best_result = (pi, m, S)
            BIC_prev = BIC

        print(k)
        k += 1
    return best_k, best_result, np.array(l), np.array(b)
