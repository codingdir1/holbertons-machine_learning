#!/usr/bin/env python3

import numpy as np

class GaussianProcess:
    def __init__(self, X_init, Y_init, l = 1, sigma_f = 1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        return (self.sigma_f ** 2) * np.exp((-1 / (2 * self.l ** 2)) * np.abs(X1 - X2.T) ** 2)

    def predict(self, X_s):
        K_s = self.kernel(X_s, self.X)
        L = np.linalg.solve(self.K, K_s.T)
        K_ss = self.kernel(X_s, X_s)
        mu = np.matmul(K_s, L)[:, 0]
        sigma = np.diag(K_ss - np.matmul(K_s, L))
        return mu, sigma

    def update(self, X_new, Y_new):
        self.X = np.concatenate((self.X, X_new[:, None]), axis = 0)
        self.Y = np.concatenate((self.Y, Y_new[:, None]), axis = 0)
        self.K = self.kernel(self.X, self.X)
