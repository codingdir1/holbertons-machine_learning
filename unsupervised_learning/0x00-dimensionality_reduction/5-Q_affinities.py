#!/usr/bin/env python3

import numpy as np

def Q_affinities(Y):
    diff = Y[:, None, :] - Y[None, :, :]
    D = np.sum(diff ** 2, axis = -1)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0.)
    total = np.sum(num)
    Q = num / total
    return (Q, num)
