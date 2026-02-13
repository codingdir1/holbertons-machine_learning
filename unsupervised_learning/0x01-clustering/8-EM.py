#!/usr/bin/env python3

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose = False):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(k, int) or k <= 0 or k > X.shape[0] or \
        not isinstance(iterations, int) or iterations <= 0 or \
        not isinstance(tol, float) or tol < 0.0 or \
        not isinstance(verbose, bool):
        return None, None, None, None

    pi, m, S = initialize(X, k)
    g, l = None, None

    for i in range(iterations):
        g, l_new = expectation(X, pi, m, S)
        pi, m, S = maximization(X, g)
        if verbose == True and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, l_new.round(5)))

        if i > 0 and abs(l - l_new) <= tol:
            if verbose == True:
                print("Log Likelihood after {} iterations: {}".format(i, l_new.round(5)))
            l = l_new
            break
        else:
            l = l_new
    
    return pi, m, S, g, l
