#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def lenet5(x, y):
    initializer = tf.keras.initializers.he_normal()

    conv1 = tf.layers.conv2D(
            inputs = x,
            filters = 6,
            kernel_size = (5, 5),
            strides = (1, 1),
            padding = "same",
            activation = tf.nn.relu,
            kernel_initializer=initializer)

    pool1 = tf.layers.MaxPooling2D(
            inputs = conv1,
            pool_size = (2, 2),
            stride = (2, 2),
            padding = "valid",
            kernel_initializer=initializer)

    conv2 = tf.layers.Conv2D(
            inputs = pool1,
            filters = 16,
            kernel_size = (5, 5),
            strides = (1, 1),
            padding = "valid",
            activation = tf.nn.relu,
            kernel_initializer=initializer)

    pool2 = tf.layers.MaxPooling2D(
            inputs = conv2,
            pool_size = (2, 2),
            stride = (2, 2),
            padding = "valid",
            kernel_initializer=initializer)

    flat = tf.layers.flatten(pool2)

    dense1 = tf.layers.dense(units = 120, 
                             activation=tf.nn.relu, 
                             kernel_initializer=initializer)(flat)

    dense2 = tf.layers.dense(units = 84, 
                             activation=tf.nn.relu, 
                             kernel_initializer=initializer)(dense1)

    dense3 = tf.layers.dense(units = 10, 
                             activation=None, 
                             kernel_initializer=initializer)(dense2)

    softmax = tf.nn.softmax(dense3)

    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y,
                                                       logits = softmax))
    
    
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return softmax, train_op, loss, accuracy
