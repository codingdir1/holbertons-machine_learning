#!/usr/bin/env python3

import numpy

class Neuron():
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = numpy.random.randn(1, self.nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
            return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
