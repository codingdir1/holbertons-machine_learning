#!/usr/bin/env python3

import numpy

def one_hot_decode(one_hot):
    try:
        return numpy.argmax(one_hot, axis = 0)
    except:
        return None
