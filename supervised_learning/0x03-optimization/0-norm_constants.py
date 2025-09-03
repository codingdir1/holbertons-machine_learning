#!/usr/bin/env python3

import numpy

def normalization_constants(X):
    return numpy.mean(X, axis = 0), numpy.var(X, axis = 0) ** 0.5
