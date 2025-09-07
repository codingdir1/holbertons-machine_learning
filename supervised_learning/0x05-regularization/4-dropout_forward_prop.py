#!/usr/bin/env python3

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        Z_i = np.matmul(weights["W{0}".format(i)], cache["A{0}".format(i - 1)]) + weights["b{0}".format(i)]
        if i < L:
            A_i = np.tanh(Z_i)
            cache["D{0}".format(i)] = (np.random.rand(A_i.shape[0], A_i.shape[1]) < keep_prob).astype(int)
            cache["A{0}".format(i)] = A_i * cache["D{0}".format(i)] / keep_prob
        else:
            cache["A{0}".format(i)] = np.exp(Z_i) / np.sum(np.exp(Z_i), axis = 0, keepdims = True)
    return cache
