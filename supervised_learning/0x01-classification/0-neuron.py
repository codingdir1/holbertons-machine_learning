#!/usr/bin/env python3

import numpy

class Neuron():
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = numpy.random.randn(1, self.nx)
        self.b = 0
        self.A = 0
