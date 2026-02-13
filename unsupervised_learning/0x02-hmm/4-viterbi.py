#!/usr/bin/env python3

import numpy as np

def viterbi(Observation, Emission, Transition, Initial):
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1 or Observation.shape[0] < 2 or \
        not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2 or \
        not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2 or \
        not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2 or \
        Emission.shape[0] != Transition.shape[0] or \
        Transition.shape[0] != Transition.shape[1] or \
        Transition.shape[1] != Initial.shape[0]:
        return None, None
    
    path = [0] * Observation.shape[0]

    deltas = np.zeros(shape = (Transition.shape[0], Observation.shape[0]))
    deltas[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    psi = np.zeros(shape = deltas.shape, dtype = int)

    for i in range(1, deltas.shape[1], 1):
        prev_states = deltas[:, i - 1] * Transition.T
        psi[:,  i] = np.argmax(prev_states, axis = 1)
        deltas[:, i] = np.max(prev_states, axis = 1) * Emission[:, Observation[i]]
        
    path[-1] = int(np.argmax(deltas[:, -1]))
    for i in range(len(path) - 2, -1, -1):
        path[i] = int(psi[path[i + 1], i + 1])

    P = np.max(deltas[:, -1])

    return path, P
