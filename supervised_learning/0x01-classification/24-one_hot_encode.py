#!/usr/bin/env python3

import numpy

def one_hot_encode(Y, classes):
    try:
        return numpy.eye(classes)[Y].T
    except:
        return None

