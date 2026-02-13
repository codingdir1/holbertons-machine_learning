#!/usr/bin/env python3

import numpy as np

def baum_welch(Observations, Transition, Emission, Initial, iterations = 1000):
    if not isinstance(Observations, np.ndarray) or len(Observations.shape) != 1 or Observations.shape[0] < 2 or \
        not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2 or \
        not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2 or \
        not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2 or \
        not isinstance(iterations, int) or iterations <= 0 or \
        Emission.shape[0] != Transition.shape[0] or \
        Transition.shape[0] != Transition.shape[1] or \
        Transition.shape[1] != Initial.shape[0]:
        return None, None
    for i in range(iterations):
        alpha = np.zeros(shape = (Transition.shape[0], Observations.shape[0]))
        alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, alpha.shape[1], 1):
            alpha[:, t] = np.matmul(alpha[:, t - 1][None, :], Transition) * Emission[:, Observations[t]]

        beta = np.zeros(shape = alpha.shape)
        beta[:, -1] = np.ones(shape = (beta.shape[0]))
        for t in range(beta.shape[1] - 2, -1, -1):
            beta[:, t] = np.matmul(Transition, beta[:, t + 1] * Emission[:, Observations[t + 1]])

        gamma = alpha * beta
        
        xi = np.zeros(shape = (Observations.shape[0], Transition.shape[0], Transition.shape[1]))
        for t in range(0, xi.shape[0] - 1, 1):
            xi[t] = alpha[:, t][:, None] * Transition * Emission[:, Observations[t + 1]] * beta[:, t + 1]
            xi[t] /= np.sum(xi[t])

        Transition = np.sum(xi, axis = 0) / np.sum(xi, axis = (0, 2))[:, None]

        gamma_filter = np.eye(Emission.shape[1])[Observations].T
        Emission = np.matmul(gamma, gamma_filter.T) / np.sum(gamma, axis = 1)[:, None]
    return Transition, Emission
