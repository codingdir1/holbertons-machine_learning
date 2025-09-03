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

        # Public instance attributes
        self.W1 = numpy.random.randn(self.nodes, self.nx)
        self.b1 = numpy.zeros((self.nodes, 1))
        self.A1 = 0
        self.W2 = numpy.random.randn(1, self.nodes)
        self.b2 = 0
        self.A2 = 0
