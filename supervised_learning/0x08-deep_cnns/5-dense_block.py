#!/usr/bin/env pyhton3

import tensorflow.keras as K

def dense_block(X, nb_filters, growth_rate, layers):

    init = K.initializers.he_normal()

    input_X = X
    prev_filters = nb_filters

    for i in range(layers):

        batch_normalization_a = K.layers.BatchNormalization(axis = -1)(input_X)
        activation_a = K.layers.Activation("relu")(batch_normalization_a)
        conv2d_a = K.layers.Conv2D(
                filters = 4 * growth_rate,
                kernel_size = (1, 1),
                strides = (1, 1),
                padding = "same",
                activation = None,
                kernel_initializer = init)(activation_a)
        
        batch_normalization_b = K.layers.BatchNormalization(axis = -1)(conv2d_a)
        activation_b = K.layers.Activation("relu")(batch_normalization_b)
        conv2d_b = K.layers.Conv2D(
                filters = growth_rate,
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = "same",
                activation = None,
                kernel_initializer = init)(activation_b)
        
        input_X = K.layers.Concatenate(axis = -1)([input_X, conv2d_b])
        prev_filters += growth_rate
    return input_X, prev_filters
