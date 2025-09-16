#!/usr/bin/env python3

import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data = None, early_stopping = False, patience = 0, learning_rate_decay = False, alpha = 0.1, decay_rate = 1, verbose = True, shuffle = False):
    early_stop = None
    if early_stopping == True and validation_data != None:
        early_stop = K.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = patience)

    if learning_rate_decay == True and validation_data != None:
        def lrs(epoch, lr):
            return lr / (1 + decay_rate * epoch)
        learning_rate_decay = K.callbacks.LearningRateScheduler(
                schedule = lrs,
                verbose = 1)

    return network.fit(
        x = data,
        y = labels,
        batch_size = batch_size,
        epochs = epochs,
        verbose = int(verbose),
        shuffle = int(shuffle),
        validation_data = validation_data,
        callbacks = [early_stop, learning_rate_decay])
