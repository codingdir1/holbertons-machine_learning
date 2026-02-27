#!/usr/bin/env python3

import numpy as np

class LSTMCell:
    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size = (h + i, h))
        self.Wu = np.random.normal(size = (h + i, h))
        self.Wc = np.random.normal(size = (h + i, h))
        self.Wo = np.random.normal(size = (h + i, h))
        self.Wy = np.random.normal(size = (h, o))
        self.bf = np.zeros(shape = (1, h))
        self.bu = np.zeros(shape = (1, h))
        self.bc = np.zeros(shape = (1, h))
        self.bo = np.zeros(shape = (1, h))
        self.by = np.zeros(shape = (1, o))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis = 1)[:, None]

    def forward(self, h_prev, c_prev, x_t):
        h = h_prev.shape[1]

        forget = self.sigmoid(np.matmul(h_prev, self.Wf[:h, :]) + np.matmul(x_t, self.Wf[h:, :]) + self.bf)
        update = self.sigmoid(np.matmul(h_prev, self.Wu[:h, :]) + np.matmul(x_t, self.Wu[h:, :]) + self.bu)
        output = self.sigmoid(np.matmul(h_prev, self.Wo[:h, :]) + np.matmul(x_t, self.Wo[h:, :]) + self.bo)
        candidate = np.tanh(np.matmul(h_prev, self.Wc[:h, :]) + np.matmul(x_t, self.Wc[h:, :]) + self.bc)
        
        c_next = c_prev * forget + update * candidate
        h_next = output * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
