#!/usr/bin/env python3

import numpy

def shuffle_data(X, Y):
    indecies = numpy.random.permutation(X.shape[0])
    return X[indecies], Y[indecies]
