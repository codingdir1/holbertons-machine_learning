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
