#!/usr/bin/env python3

import tensorflow.keras as K

def save_model(network, filename):
    network.save(filepath = filename)
    return None

def load_model(filename):
    return  K.models.load_model(filepath = filename)