#!/usr/bin/env python3

import tensorflow.keras as K

def save_config(network, filename):
    json_config = network.to_json()
    f = open(filename, "w")
    f.write(json_config)
    f.close()
    return None

def load_config(filename):
    f = open(filename, "r")
    json_config = f.read()
    f.close()
    return K.models.model_from_json(json_config)
