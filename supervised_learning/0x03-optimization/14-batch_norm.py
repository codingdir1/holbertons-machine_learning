#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def create_batch_norm_layer(prev, n, activation):
    gamma = tf.Variable(tf.ones(shape =  (1, n)), trainable = True)
    beta = tf.Variable(tf.zeros(shape = (1, n)), trainable = True)
    epsilon = 1e-8

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
            units = n,
            kernel_initializer = initializer)(prev)

    mean, var = tf.nn.moments(layer, axes = [0])
    batch_norm = tf.nn.batch_normalization(
        x = layer,
        mean = mean,
        variance = var,
        offset = beta,
        scale = gamma,
        variance_epsilon = epsilon)

    return activation(batch_norm)
