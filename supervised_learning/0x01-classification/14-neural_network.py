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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        self.__W1 -= alpha * numpy.matmul(numpy.matmul(self.__W2.T, A2 - Y) * (A1 * (1 - A1)), X.T) / Y.shape[1]
        self.__b1 -= alpha * numpy.sum(numpy.matmul(self.__W2.T, A2 - Y) * (A1 * (1 - A1)), axis = 1, keepdims = True) / Y.shape[1]
        self.__W2 -= alpha * numpy.matmul(A2 - Y, A1.T) / Y.shape[1]
        self.__b2 -= alpha * numpy.sum(A2 - Y, axis = 1, keepdims = True) / Y.shape[1]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be positive integer")
        elif type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be a positive float")

        i = 0
        while i <= iterations:
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            i += 1
        return self.evaluate(X, Y)
