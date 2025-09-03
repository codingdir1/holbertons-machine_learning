#!/usr/bin/env python3

import numpy

class NeuralNetwork:
    def __init__(self, nx, nodes):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes

        # Private instance attributes
        self.__W1 = numpy.random.randn(self.nodes, self.nx)
        self.__b1 = numpy.zeros((self.nodes, 1))
        self.__A1 = 0
        self.__W2 = numpy.random.randn(1, self.nodes)
        self.__b2 = 0
        self.__A2 = 0

    # getter functions

    @property
    def W1(self):
        return self.__W1
        
    @property
    def b1(self):
        return self.__b1
    @property
    def A1(self):
        return self.__A1
    @property
    def W2(self):
        return self.__W2
    @property
    def b2(self):
        return self.__b2
    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        self.__A1 = 1 / (1 + numpy.exp(-1 * (numpy.matmul(self.__W1, X) + self.b1)))
        self.__A2 = 1 / (1 + numpy.exp(-1 * (numpy.matmul(self.__W2, self.__A1) + self.b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        return -1 * numpy.sum(Y * numpy.log(A) + (1 - Y) * numpy.log(1.0000001 - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        self.forward_prop(X)
        return numpy.where(self.__A2 >= 0.5, 1, 0), self.cost(Y, self.__A2)
