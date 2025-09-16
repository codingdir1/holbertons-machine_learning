#!/usr/bin/env python3

import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    
    A_prev_pad = A_prev

    if padding == "same":
        # output dimensions
        pad_h = (W.shape[0] - 1) // 2
        pad_w = (W.shape[1] - 1) // 2

        output_h = (A_prev.shape[1] + 2 * pad_h - W.shape[0]) // stride[0] + 1
        output_w = (A_prev.shape[2] + 2 * pad_w - W.shape[1]) // stride[1] + 1

        # create the output layer
        output = np.zeros(shape = (A_prev.shape[0], output_h, output_w, W.shape[3]))
        pad = (max((output_h - 1) * stride[0] + W.shape[0] - A_prev.shape[1], 0),
               max((output_w - 1) * stride[1] + W.shape[1] - A_prev.shape[2], 0))
        A_prev_pad = np.pad(array = A_prev, 
                            pad_width = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]), (0, 0)),
                            mode = "constant",
                            constant_values = 0)
    # calculate dA_prev
    dA_prev = np.zeros(shape = A_prev_pad.shape)

    # calculate dW
    dW = np.zeros(shape = W.shape)
    for m in range(dZ.shape[0]):
        for i in range(dZ.shape[1]):
            y = i * stride[0]
            for j in range(dZ.shape[2]):
                x = j * stride[1]
                A_slice = A_prev_pad[m, y : y + W.shape[0], x : x + W.shape[1], :]
                for c in range(dZ.shape[3]):
                    dW[..., c] += A_slice * dZ[m, i, j, c]
                    dA_prev[m, y : y + W.shape[0], x : x + W.shape[1], :] += W[..., c] * dZ[m, i, j, c]

    # calculate db
    db = np.sum(dZ, axis = (0, 1, 2), keepdims = True)

    return dA_prev, dW, db
