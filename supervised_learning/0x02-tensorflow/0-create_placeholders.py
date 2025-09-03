#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

def create_placeholders(nx, classes):
    x = tf.placeholder(shape=(None, nx), dtype = "float32", name = "x")
    y = tf.placeholder(shape=(None, classes), dtype = "float32", name = "y")
    return x, y
