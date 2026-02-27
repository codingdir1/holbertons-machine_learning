#!/usr/bin/env python3

import numpy as np

class GRUCell:
    def __init__(self, i, h, o):
        self.Wz = np.random.normal(size = (h + i, h))
        self.Wr = np.random.normal(size = (h + i, h))
        self.Wh = np.random.normal(size = (h + i, h))
        self.Wy = np.random.normal(size = (h, o))
        self.bz = np.zeros(shape = (1, h))
        self.br = np.zeros(shape = (1, h))
        self.bh = np.zeros(shape = (1, h))
        self.by = np.zeros(shape = (1, o))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis = 1)[:, None]

    def forward(self, h_prev, x_t):
        h = h_prev.shape[1]
        update = self.sigmoid(np.matmul(h_prev, self.Wz[:h, :]) + np.matmul(x_t, self.Wz[h:, :]) + self.bz)
        reset = self.sigmoid(np.matmul(h_prev, self.Wr[:h, :]) + np.matmul(x_t, self.Wr[h:, :]) + self.br)
        candidate = np.tanh(np.matmul(reset * h_prev, self.Wh[:h, :]) + np.matmul(x_t, self.Wh[h:, :]) + self.bh)

        h_next = (1 - update) * h_prev + update * candidate
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
