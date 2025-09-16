#!/usr/bin/env python3

import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, verbose = True, shuffle = False):
    return network.fit(
            x = data,
            y = labels,
            batch_size = batch_size,
            epochs = epochs,
            verbose = 1 if verbose else 0,
            shuffle = shuffle)
