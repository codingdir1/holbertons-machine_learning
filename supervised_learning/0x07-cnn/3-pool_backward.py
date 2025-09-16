#!/usr/bin/env python3

import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride = (1, 1), mode = "max"):
    dA_prev = np.zeros(shape = A_prev.shape)

    for m in range(dA.shape[0]):
        for i in range(dA.shape[1]):
            y = i * stride[0]
            for j in range(dA.shape[2]):
                x = j * stride[1]
                for c in range(dA.shape[3]):
                    if mode == "max":
                        A_slice = A_prev[m, y : y + kernel_shape[0], x : x + kernel_shape[1], c]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[m, y : y + kernel_shape[0], x : x + kernel_shape[1], c] += mask * dA[m, i, j, c]

                    elif mode == "avg":
                        avg = dA[m, i, j, c] // (kernel_shape[0] * kernel_shape[1])
                        dA_prev[m, y : y + kernel_shape[0], x : x + kernel_shape[1], c] += avg
    return dA_prev