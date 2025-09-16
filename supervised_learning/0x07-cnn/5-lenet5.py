#!/usr/bin/env python3

import tensorflow.keras as K

def lenet5(X):
    initializer = K.initializers.he_normal()

    model = K.Sequential([
        K.layers.Conv2D(filters = 6,
                        kernel_size = (5, 5),
                        strides = (1, 1),
                        padding = "same",
                        activation = "relu",
                        input_shape = (28, 28, 1),
                        kernel_initializer = initializer),
        K.layers.MaxPooling2D(
                        pool_size = (2, 2),
                        strides = (2, 2),
                        padding="valid"),
        K.layers.Conv2D(filters = 16,
                        kernel_size = (5, 5),
                        strides = (1, 1),
                        padding = "valid",
                        activation = "relu",
                        kernel_initializer = initializer),
        K.layers.MaxPooling2D(
                        pool_size = (2, 2),
                        strides = (2, 2),
                        padding="valid"),
        K.layers.Flatten(),
        K.layers.Dense(
            units = 120,
            activation = "relu",
            kernel_initializer = initializer),
        K.layers.Dense(
            units = 84,
            activation = "relu",
            kernel_initializer = initializer),
        K.layers.Dense(
            units = 10,
            activation = "softmax",
            kernel_initializer = initializer)
        ])

    optimizer = K.optimizers.Adam()

    model.compile(optimizer = optimizer,
                  loss = "categorical_crossentropy",
                  metrics = ["accuracy"])
    return model


