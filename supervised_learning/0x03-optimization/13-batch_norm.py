#!/usr/bin/env python3

import numpy

def batch_norm(Z, gamma, beta, epsilon):
    mean = numpy.mean(Z, axis = 0, keepdims = True)
    std = numpy.std(Z, axis = 0, keepdims = True)
    Z_norm = (Z - mean) / (std ** 2 + epsilon) ** 0.5
    return gamma * Z_norm + beta
