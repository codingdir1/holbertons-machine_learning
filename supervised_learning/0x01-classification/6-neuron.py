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

    def forward_prop(self, X):
        self.__A = 1 / (1 + numpy.exp(-1 * (numpy.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        cost_matrix = Y * numpy.log(A) + (1 - Y) * numpy.log(1.0000001 - A)
        return -1 * numpy.sum(cost_matrix) / Y.shape[1]

    def evaluate(self, X, Y):
        self.forward_prop(X)
        return (numpy.where(self.__A >= 0.5, 1, 0), self.cost(Y, self.__A))

    def gradient_descent(self, X, Y, A, alpha = 0.05):
        self.__W -= alpha * numpy.matmul(A - Y, X.T) / Y.shape[1]
        self.__b -= alpha * numpy.sum(A - Y) / Y.shape[1]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if (type(iterations) != int):
            raise TypeError("iterations must be an integer")
        elif (iterations <= 0):
            raise TypeError("iterations must be a positive integer")
        elif (type(alpha) != float):
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        i = 0
        while i < iterations:
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            i += 1
        return self.evaluate(X, Y)
