#!/usr/bin/env python3

import tensorflow.keras as K

def save_weights(network, filename, save_format = 'h5'):
    if filename[-3:] != "." + save_format:
        filename += "." + save_format
    network.save_weights(filepath = filename)

def load_weights(network, filename):
    network.load_weights(filename)
