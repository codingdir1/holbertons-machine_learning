#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess

class BayesianOptimization:
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l = 1, sigma_f = 1, xsi = 0.01, minimize = True):
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)[:, None]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        f_opt = None
        mu, sig = self.gp.predict(self.gp.X)
        sample_mu, sample_sig = self.gp.predict(self.X_s)
        sample_sig

        if self.minimize ==  True:
            f_opt = np.min(mu)
            imp = f_opt - sample_mu - self.xsi
        else:
            f_opt = np.max(mu)
            imp = sample_mu - f_opt - self.xsi

        Z = imp / (sample_sig + 1e-9)
        EI = imp * norm.cdf(Z) + sample_sig * norm.pdf(Z)
        EI[sample_sig <= 1e-9] = 0.0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
