#!/usr/bin/env python3

import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    for i in range(L, 0, -1):
        if i == L:
            Dcost_by_Dz = cache["A{0}".format(i)] - Y
        else:
            Dcost_by_Dz = np.matmul(W_prev.T, Dcost_by_Dz) * (1 - cache["A{0}".format(i)] ** 2) * cache["D{0}".format(i)] / keep_prob
        W_prev = np.copy(weights["W{0}".format(i)])
        Dcost_by_Dw = np.matmul(Dcost_by_Dz, cache["A{0}".format(i - 1)].T)

        weights["W{0}".format(i)] -= alpha * Dcost_by_Dw / Y.shape[1]
        weights["b{0}".format(i)] -= np.sum(alpha * Dcost_by_Dz, keepdims = True, axis = 1) / Y.shape[1]

    return weights
