#!/usr/bin/env python3

import numpy as np

def deep_rnn(rnn_cells, X, h_0):
    H, Y = [h_0], []
    for t in range(X.shape[0]):
        H_prev = H[-1]
        H_T = []
        for l in range(len(rnn_cells)):
            h_prev = H_prev[l]
            if l == 0:
                cell_input = X[t]
            else:
                cell_input = h_next
            h_next, y = rnn_cells[l].forward(h_prev, cell_input)
            H_T.append(h_next)
        H.append(np.array(H_T))
        Y.append(y)
    return np.array(H), np.array(Y)
