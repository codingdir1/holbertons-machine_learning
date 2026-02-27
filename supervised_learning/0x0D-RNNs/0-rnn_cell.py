#!/usr/bin/env python3

import numpy as np

class RNNCell:
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size = (h + i, h))
        self.Wy = np.random.normal(size = (h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        exp_n = np.exp(x)
        return exp_n / np.sum(exp_n, axis = 1)[:, None]

    def forward(self, h_prev, x_t):
        h = h_prev.shape[1]
        h_next = np.tanh(np.matmul(h_prev, self.Wh[: h, :]) + np.matmul(x_t, self.Wh[h :, :]) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y

