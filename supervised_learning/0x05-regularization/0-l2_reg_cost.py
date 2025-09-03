#!/usr/bin/env python3

import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    summation = 0
    for i in range(1, L + 1):
        summation += np.sum(weights["W{0}".format(i)] ** 2)
    l2_reg = 0.5 * summation

    return cost + lambtha * l2_reg / m
