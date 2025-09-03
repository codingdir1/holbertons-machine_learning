#!/usr/bin/env python3

import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):

    for i in range(L, 0, -1):
        if i == L:
            d_cost_by_d_z = (cache["A{0}".format(L)] - Y)
        elif i < L:
            d_cost_by_d_z = np.matmul(W_prev.T, d_cost_by_d_z) * (1 - cache["A{0}".format(i)] ** 2)

        d_cost_by_d_w = (np.matmul(d_cost_by_d_z, cache["A{0}".format(i - 1)].T) + lambtha * weights["W{0}".format(i)]) / Y.shape[0]
        d_cost_by_d_b = np.sum(d_cost_by_d_z, axis = 1, keepdims = True) / Y.shape[0]

        W_prev = np.copy(weights["W{0}".format(i)])

        weights["W{0}".format(i)] -= alpha * d_cost_by_d_w
        weights["b{0}".format(i)] -= alpha * d_cost_by_d_b
