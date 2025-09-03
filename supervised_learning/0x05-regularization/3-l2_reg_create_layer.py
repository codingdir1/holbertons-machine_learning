#!/usr/bin/env python3
 
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def l2_reg_create_layer(prev, n, activation, lambtha):
    init = tf.keras.initializers.VarianceScaling(scale = 2.0, mode = "fan_avg")
    reg = tf.keras.regularizers.L2(l2 = lambtha)

    return tf.keras.layers.Dense(
            units = n,
            activation = activation,
            kernel_initializer = init,
            kernel_regularizer = reg)(prev)
