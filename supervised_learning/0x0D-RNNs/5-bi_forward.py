#!/usr/bin/env python3

import numpy as np

class BidirectionalCell:
    def __init__(self, i, h, o):
        self.Whf = np.random.normal(size = (h + i, h))
        self.Whb = np.random.normal(size = (h + i, h))
        self.Wy = np.random.normal(size = (2 * h, o))
        self.bhf = np.zeros(shape = (1, h))
        self.bhb = np.zeros(shape = (1, h))
        self.by = np.zeros(shape = (1, o))

    @staticmethod
    def softmax(z):
        exp_z = exp_z = np.exp(z - np.max(z, axis = 1)[:, None])
        return exp_z / np.sum(exp_z , axis = 1)[:, None]

    def forward(self, h_prev, x_t):
        h = h_prev.shape[1]
        return np.tanh(np.matmul(h_prev, self.Whf[:h, :]) + np.matmul(x_t, self.Whf[h:, :]) + self.bhf)
