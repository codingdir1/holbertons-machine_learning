#!/usr/bin/env python3

import numpy as np

def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    output_h = (A_prev.shape[1] - kernel_shape[0]) // stride[0] + 1
    output_w = (A_prev.shape[2] - kernel_shape[1]) // stride[1] + 1

    output = np.ndarray(shape = (A_prev.shape[0], output_h, output_w, A_prev.shape[3]), 
                        dtype = "float32")

    for i in range(output_h):
        for j in range(output_w):
            y = i * stride[0]
            x = j * stride[1]
            for c in range(output.shape[3]):
                if mode == "max":
                    output[:, i, j, c] = np.max(A_prev[:, y : y + kernel_shape[0], x : x + kernel_shape[1], c],
                                                axis = (1, 2))
                elif mode == "avg":
                    output[:, i, j, c] = np.avg(A_prev[:, y : y + kernel_shape[0], x : x + kernel_shape[1], c],
                                                axis = (1, 2))

    return output
