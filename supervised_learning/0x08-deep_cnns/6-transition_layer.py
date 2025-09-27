#!/usr/bin/env python3

import tensorflow.keras as K

def transition_layer(X, nb_filters, compression):

    init = K.initializers.he_normal()

    batch_normalization = K.layers.BatchNormalization(axis = -1)(X)
    activation = K.layers.Activation("relu")(batch_normalization)
    conv2d = K.layers.Conv2D(
            filters = int(nb_filters * compression),
            kernel_size = (1, 1),
            strides = (1, 1),
            padding = "same",
            activation = None,
            kernel_initializer = init)(activation)
    average_pooling2d = K.layers.AveragePooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
            padding = "valid")(conv2d)

    return average_pooling2d, average_pooling2d.shape[-1]
