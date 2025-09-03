#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes = [], activations = []):
    prev = x
    for i in range(len(layer_sizes)):
        layer = create_layer(prev, layer_sizes[i], activations[i])
        prev = layer
    return layer

