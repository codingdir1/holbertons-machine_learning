#!/usr/bin/env python3

import numpy as np

def forward(Observation, Emission, Transition, Initial):
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1 or Observation.shape[0] < 2 or \
        not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2 or \
        not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2 or \
        not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2 or \
        Emission.shape[0] != Transition.shape[0] or \
        Transition.shape[0] != Transition.shape[1] or \
        Transition.shape[1] != Initial.shape[0]:
        return None, None

    F = np.zeros(shape = (Emission.shape[0] , Observation.shape[0]))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for i in range(1, F.shape[1], 1):
        F[:, i] = np.matmul(F[:, i - 1][None, :], Transition)[0, :] * Emission[:, Observation[i]]
        
    P = np.sum(F[:, -1])

    return P, F
