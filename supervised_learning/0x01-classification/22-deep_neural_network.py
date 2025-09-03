#!/usr/bin/env python3

import numpy

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx <= 0:
            raise ValueError("nx must be a positive integer")
        elif type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        elif not numpy.issubdtype(numpy.array(layers).dtype, numpy.integer):
            raise ValueError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.__L = len(self.layers)
        self.__cache = {}
        self.__weights = {}
        
        i = 1
        fan_in = self.nx
        while i <= self.__L:
            self.__weights["W{0}".format(i)] = numpy.random.randn(self.layers[i - 1], fan_in) * numpy.sqrt(2 /fan_in)
            self.__weights["b{0}".format(i)] = numpy.zeros((self.layers[i - 1], 1))
            fan_in = self.layers[i - 1]
            i += 1
    
    @property
    def L(self):
        return self.__L
    
    @property
    def cache(self):
        return self.__cache
    
    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache["A0"] = X
        i = 1
        input_of_layer = X
        while i <= self.__L:
            self.__cache["A{0}".format(i)] = 1 / (1 + numpy.exp(-1 * (numpy.matmul(self.__weights["W{0}".format(i)], input_of_layer) + self.__weights["b{0}".format(i)])))
            input_of_layer = self.__cache["A{0}".format(i)]
            i += 1
        return self.__cache["A{0}".format(i - 1)], self.__cache

    def cost(self, Y, A):
        return -1 * numpy.sum(Y * numpy.log(A) + (1 - Y) * numpy.log(1.0000001 - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        self.forward_prop(X)
        A = self.__cache["A{0}".format(self.__L)]
        return numpy.where(A >= 0.5, 1, 0), self.cost(Y, A)
    
    def gradient_descent(self, Y, cache, alpha = 0.05):
        i = self.__L
        while i >= 1:
            if i == self.__L:
                Dcost_by_Dz = (cache["A{0}".format(i)] - Y)
                W_prev = numpy.copy(self.__weights["W{0}".format(i)])
            else:
                Dcost_by_Dz = numpy.matmul(W_prev.T, Dcost_by_Dz) * (cache["A{0}".format(i)] * (1 - cache["A{0}".format(i)]))
                W_prev = numpy.copy(self.__weights["W{0}".format(i)])
            Dcost_by_Dw = numpy.matmul(Dcost_by_Dz, cache["A{0}".format(i - 1)].T)
            self.__weights["W{0}".format(i)] -= alpha * Dcost_by_Dw / Y.shape[1]
            self.__weights["b{0}".format(i)] -= alpha * numpy.sum(Dcost_by_Dz, keepdims = True, axis = 1) / Y.shape[1]
            i -= 1
    
    def train(self, X, Y, iterations = 5000, alpha = 0.05):
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif type(alpha) != float:
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        i = 0
        while i < iterations:
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache)
            i += 1
        return self.evaluate(X, Y)
