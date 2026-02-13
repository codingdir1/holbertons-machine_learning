#!/usr/bin/env python3

import numpy as np

def markov_chain(P, s, t = 1):
    if not isinstance(P, np.ndarray) or len(P.shape) != 2 or P.shape[0] != P.shape[1] or \
        not isinstance(s, np.ndarray) or len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[1] or \
        not isinstance(t, int) or t <= 0:
        return None

    P_state = s
    for i in range(t):
        P_state = np.matmul(P_state, P)
    return P_state
