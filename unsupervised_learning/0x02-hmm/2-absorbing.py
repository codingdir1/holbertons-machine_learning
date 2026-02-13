#!/usr/bin/env python3

import numpy as np

def absorbing(P):
    if not isinstance(P, np.ndarray) or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False

    absorbing = np.diag(P) == 1.0
    if np.all(absorbing == False):
        return False
    else:
        transient = ~absorbing
        Q = P[transient][:, transient]
        eig_vals = np.linalg.eigvals(Q)
        if np.all(np.abs(eig_vals) < 1 - 1e-9):
            return True
        return False
