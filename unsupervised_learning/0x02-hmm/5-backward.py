#!/usr/bin/env python3

import numpy as np

def backward(Observation, Emission, Transition, Initial):
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1 or Observation.shape[0] < 2 or \
        not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2 or \
        not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2 or \
        not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2 or \
        Emission.shape[0] != Transition.shape[0] or \
        Transition.shape[0] != Transition.shape[1] or \
        Transition.shape[1] != Initial.shape[0]:
        return None, None

    B = np.zeros(shape = (Emission.shape[0], Observation.shape[0]))
    B[:, -1] = np.ones(shape = (Transition.shape[0],))

    for i in range(B.shape[1] - 1, 0, -1):
        B[:, i - 1] = np.matmul(Transition, (Emission[:, Observation[i]] * B[:, i]))

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
