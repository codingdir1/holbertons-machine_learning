#!/usr/bin/env python3

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    reg = K.regularizers.l2(lambtha)

    input_layer = K.Input(shape = (nx, ))
    layer = K.layers.Dense(units = layers[0],
                           activation = activations[0],
                           kernel_regularizer = reg)(input_layer)

    for i in range(1, len(layers)):
        layer = K.layers.Dropout(rate = 1 - keep_prob)(layer)
        layer = K.layers.Dense(units = layers[i],
                               activation = activations[i],
                               kernel_regularizer = reg)(layer)

    return K.Model(inputs = input_layer, outputs = layer)
