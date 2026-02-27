#!/usr/bin/env python3

import numpy as np

def rnn(rnn_cell, X, h_0):
    h_prev = h_0
    H, Y = [h_prev], []
    for x_t in X:
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H.append(h_next)
        Y.append(y)
        h_prev = h_next
    return np.array(H), np.array(Y)
