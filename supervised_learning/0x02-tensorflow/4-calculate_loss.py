#!usr/bin/env python3

import tensorflow.compat.v1 as tf

def calculate_loss(y, y_pred):
    return tf.losses.softmax_cross_entropy(onehot_labels = y, logits = y_pred)
