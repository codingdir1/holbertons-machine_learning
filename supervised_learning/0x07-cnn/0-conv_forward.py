#!/usr/bin/env python3

import numpy as np

def conv_forward(A_prev, W, b, activation, padding = "same", stride = (1, 1)):
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
        
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                __i = i * stride[0]
                __j = j * stride[1]
                for c in range(output.shape[3]):
                    output[:, i, j, c] = np.sum(A_prev_pad[:, __i : __i + W.shape[0], __j : __j + W.shape[1], :] * W[..., c], axis = (1, 2, 3))
        output += b
        return activation(output)
    elif padding == "valid":

        # output dimensions
        output_h =  (A_prev.shape[1] - W.shape[0]) // stride[0] + 1
        output_w =  (A_prev.shape[2] - W.shape[1]) // stride[1] + 1

        # create the output layer
        output = np.zeros(shape = (A_prev.shape[0], output_h, output_w, W.shape[3]))
        
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                __i = i * stride[0]
                __j = j * stride[1]
                for c in range(output.shape[3]):
                    output[:, i, j, c] += np.sum(A_prev[:, __i : __i + W.shape[0], __j : __j + W.shape[1], :] * W[..., c], axis = (1, 2, 3))

        output += b
        return activation(output)
    else:
        raise ValueError("padding must be either same or valid")
