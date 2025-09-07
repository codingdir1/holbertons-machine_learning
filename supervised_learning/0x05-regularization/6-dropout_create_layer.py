#!#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()   # turns off TF2 and Keras 3 APIs, restores TF1 behavior

def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    # Dense layer (now available again under compat.v1)
    layer = tf.layers.dense(inputs=prev, units=n, activation=activation)

    # Dropout (rate = drop probability, not keep probability)
    layer = tf.layers.dropout(inputs=layer,
                              rate=1 - keep_prob,
                              training=training)
    return layer