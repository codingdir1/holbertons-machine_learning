#!/usr/bin/env python3

import numpy as np

pdf = __import__('5-pdf').pdf

def expectation(X, pi, m, S):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
        not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or \
        not isinstance(m, np.ndarray) or len(m.shape) != 2 or \
        not isinstance(S, np.ndarray) or len(S.shape) != 3 or \
        not (X.shape[1] == m.shape[1] == S.shape[1] == S.shape[2]) or \
        not (pi.shape[0] == m.shape[0] == S.shape[0]):
        return None, None

    g = []
    for i in range(pi.shape[0]):
        g.append(pdf(X, m[i], S[i]) * pi[i])
    g = np.array(g)
    marginal = np.sum(g, axis = 0)
    g = g / marginal
    l = np.sum(np.log(marginal))
    return g, l
