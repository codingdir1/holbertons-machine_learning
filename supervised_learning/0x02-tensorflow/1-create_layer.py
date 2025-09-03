#!/usr/bin/env python3

import tensorflow.compat.v2 as tf

def create_layer(prev, n, activation):
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units = n, activation = activation, kernel_initializer = initializer, name = "layer")(prev)
    return layer
