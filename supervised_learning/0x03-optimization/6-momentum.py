#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def create_momentum_op(loss, alpha, beta1):
    optimizer = tf.train.MomentumOptimizer(
            learning_rate = alpha,
            momentum = beta1)
    train_op = optimizer.minimize(loss)
    return train_op
