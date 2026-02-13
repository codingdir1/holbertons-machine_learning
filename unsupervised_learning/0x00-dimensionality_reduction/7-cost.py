#!/usr//bin/env python3

import numpy as np

def cost(P, Q):
    C = np.sum(P * np.log((P + 1e-12) / (Q + 1e-12)))
    return C
