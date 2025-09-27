#!/usr/bin/env python3

import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer

def densenet121(growth_rate = 32, compression = 1.0):
    init = K.initializers.he_normal()

    X = K.layers.Input(shape = (224, 224, 3))

    batch_normalization = K.layers.BatchNormalization(axis = -1)(X)
    activation = K.layers.Activation("relu")(batch_normalization)
    conv2d = K.layers.Conv2D(
            filters = 2 * growth_rate,
            kernel_size = (7, 7),
            strides = (2, 2),
            activation = None,
            kernel_initializer = init,
            padding = "same")(activation)
    max_pooling2d = K.layers.MaxPooling2D(
            pool_size = (3, 3),
            strides = (2, 2),
            padding = "same")(conv2d)

    nb_filters = max_pooling2d.shape[-1]

    dense_block_1, nb_filters = dense_block(max_pooling2d, nb_filters, growth_rate, 6)
    transition_layer_1, nb_filters = transition_layer(dense_block_1, nb_filters, compression)

    dense_block_2, nb_filters = dense_block(transition_layer_1, nb_filters, growth_rate, 12)
    transition_layer_2, nb_filters = transition_layer(dense_block_2, nb_filters, compression)

    dense_block_3, nb_filters = dense_block(transition_layer_2, nb_filters, growth_rate, 24)
    transition_layer_3, nb_filters = transition_layer(dense_block_3, nb_filters, compression)

    dense_block_4, nb_filters = dense_block(transition_layer_3, nb_filters, growth_rate, 16)

    average_pooling2d = K.layers.AveragePooling2D(
            pool_size = (7, 7),
            strides = (7, 7),
            padding="valid")(dense_block_4)

    dense = dense = K.layers.Dense(
            units = 1000,
            activation = "softmax",
            kernel_initializer = init)(average_pooling2d)

    return K.Model(inputs = X, outputs = dense)
