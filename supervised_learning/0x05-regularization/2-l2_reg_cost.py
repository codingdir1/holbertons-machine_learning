#!/usr/bin/env python3

import tensorflow.compat.v1 as tf

def l2_reg_cost(cost):
    reg_losses = tf.losses.get_regularization_losses()
    if reg_losses:  # if not empty
        return cost + tf.add_n(reg_losses)
    return cost
