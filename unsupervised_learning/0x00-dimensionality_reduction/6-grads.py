#!/usr/bin/env python3

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities

def grads(Y, P):
    Q, num = Q_affinities(Y)
    A = (P - Q) * num
    dY = (np.sum(A, axis = 1, keepdims = True) * Y) - np.matmul(A, Y)
    return (dY, Q)
