#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    optimizer = tf.train.AdamOptimizer(
            learning_rate = alpha,
            beta1 = beta1,
            beta2 = beta2,
            epsilon = epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
