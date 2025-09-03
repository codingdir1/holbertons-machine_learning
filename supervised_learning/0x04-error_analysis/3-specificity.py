#!/usr/bin/env python3 

import numpy as np

def specificity(confusion):
    result = np.ndarray(shape = (confusion.shape[0]), dtype = "float32")
    for i in range(10):
        TN_FP = np.delete(confusion, i, axis = 0)
        TN = np.delete(TN_FP, i, axis = 1)
        result[i] = np.sum(TN) / np.sum(TN_FP)
    return result

