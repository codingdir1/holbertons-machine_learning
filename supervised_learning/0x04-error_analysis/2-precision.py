#!/usr/bin/env python3

import numpy as np

def precision(confusion):
    return np.diag(confusion) / np.sum(confusion, axis = 0)
