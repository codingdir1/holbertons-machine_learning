#!/usr/bin/env python3

import numpy as np

def regular(P):
    if not isinstance(P, np.ndarray) or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    eig_vals, eig_vecs = np.linalg.eig(P.T)

    eig_val_one = np.abs(eig_vals - 1.0) < 1e-9
    if np.sum(eig_val_one) != 1:
        return None

    eig_val_not_one = eig_vals[~eig_val_one]
    if np.sum(eig_val_not_one >= 1.0) != 0:
        return None

    i = np.argmax(eig_val_one)
    stable_state = np.real(np.abs(eig_vecs[:, i].T))
    stable_state /= np.sum(stable_state)
    return stable_state
