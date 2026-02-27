#!/usr/bin/env python3

import numpy as np

def bi_rnn(bi_cell, X, h_0, h_t):
    H = []
    for t in range(X.shape[0]):
        h_next = bi_cell.forward(h_0, X[t])
        h_prev = bi_cell.forward(h_t, X[t])
        H.append(np.concatenate((h_next, h_prev), axis = -1))
    H = np.array(H)
    return H, bi_cell.output(H)
