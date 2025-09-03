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
        self.L = len(self.layers)
        self.cache = {}
        self.weights = {}
        
        i = 0
        fan_in = self.nx
        while i < self.L:
            self.weights["W{0}".format(i + 1)] = numpy.random.randn(self.layers[i], fan_in) * numpy.sqrt(2 /fan_in)
            self.weights["b{0}".format(i + 1)] = numpy.zeros((self.layers[i], 1))
            fan_in = self.layers[i]
            i += 1
