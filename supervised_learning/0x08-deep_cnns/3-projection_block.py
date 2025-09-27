#!/usr/bin/env python3

import tensorflow.keras as K

def projection_block(A_prev, filters, s=2):

    init = K.initializers.he_normal()

    conv2d = K.layers.Conv2D(
                filters = filters[0],
                kernel_size = (1, 1),
                strides = (s, s),
                padding = "same",
                kernel_initializer = init)(A_prev)
    
    batch_norm = K.layers.BatchNormalization()(conv2d)

    activation = K.layers.Activation("relu")(batch_norm)
    
    conv2d_1 = K.layers.Conv2D(
                filters = filters[1],
                kernel_size = (3, 3),
                strides = (1, 1),
                padding = "same",
                kernel_initializer = init)(activation)
    
    batch_norm_1 = K.layers.BatchNormalization(axis = -1)(conv2d_1)

    activation_1 = K.layers.Activation("relu")(batch_norm_1)
    
    conv2d_2 = K.layers.Conv2D(
                filters = filters[2],
                kernel_size = (1, 1),
                strides = (1, 1),
                padding = "same",
                kernel_initializer = init)(activation_1)

    batch_norm_2 = K.layers.BatchNormalization(axis = -1)(conv2d_2)

    conv2d_3 = K.layers.Conv2D(
                filters = filters[2],
                kernel_size = (1, 1),
                strides = (s, s),
                padding = "same",
                kernel_initializer = init)(A_prev)
    
    batch_norm_3 = K.layers.BatchNormalization(axis = -1)(conv2d_3)


    activation_2 = K.layers.Activation("relu")(batch_norm_2 + batch_norm_3)

    return activation_2

