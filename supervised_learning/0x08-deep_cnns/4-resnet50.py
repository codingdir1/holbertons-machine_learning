#!/usr/bin/env python3

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block

def resnet50():
    
    init = K.initializers.he_normal()

    X = K.layers.Input(shape = (224, 224, 3))

    conv2d = K.layers.Conv2D(
                filters = 64,
                kernel_size = (7, 7),
                strides = (2, 2),
                padding = "same",
                activation = None,
                kernel_initializer = init)(X)

    batch_normalization = K.layers.BatchNormalization(axis = -1)(conv2d)

    activation = K.layers.Activation("relu")(batch_normalization)

    max_pooling2d = K.layers.MaxPooling2D(
                pool_size = (3, 3),
                strides = (2, 2),
                padding="same")(activation)

    projection = projection_block(max_pooling2d, (64, 64, 256), s = 1)
    identity = identity_block(projection, (64, 64, 256))
    identity_1 = identity_block(identity, (64, 64, 256))

    projection_1 = projection_block(identity_1, (128, 128, 512))
    identity_2 = identity_block(projection_1, (128, 128, 512))
    identity_3 = identity_block(identity_2, (128, 128, 512))
    identity_4 = identity_block(identity_3,(128, 128, 512))

    projection_2 = projection_block(identity_4, (256, 256, 1024))
    identity_5 = identity_block(projection_2,(256, 256, 1024))
    identity_6 = identity_block(identity_5, (256, 256, 1024))
    identity_7 = identity_block(identity_6, (256, 256, 1024))
    identity_8 = identity_block(identity_7, (256, 256, 1024))
    identity_9 = identity_block(identity_8, (256, 256, 1024))
    
    projection_3 = projection_block(identity_9, (512, 512, 2048))
    identity_10 = identity_block(projection_3, (512, 512, 2048))
    identity_11 = identity_block(identity_10,(512, 512, 2048))

    average_pooling2d = K.layers.AveragePooling2D(
            pool_size = (7, 7),
            strides = (7, 7),
            padding="valid")(identity_11)

    dense = dense = K.layers.Dense(
            units = 1000,
            activation = "softmax",
            kernel_initializer = init)(average_pooling2d)

    return K.Model(inputs = X, outputs = dense)
