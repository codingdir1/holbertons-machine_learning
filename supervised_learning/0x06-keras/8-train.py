#!/usr/bin/env python3

import tensorflow.keras as K

def train_model(network, data, labels, batch_size, epochs, validation_data = None, early_stopping = False, patience = 0, learning_rate_decay = False, alpha = 0.1, decay_rate = 1, save_best = False, filepath = None, verbose = True, shuffle = False):
    callbacks = []
    if early_stopping == True and validation_data != None:
        early_stop = K.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = patience)
        callbacks.append(early_stop)

    if learning_rate_decay == True and validation_data != None:
        def lrs(epoch):
            return alpha / (1 + decay_rate * epoch)
        learning_rate_decay = K.callbacks.LearningRateScheduler(
                schedule = lrs,
                verbose = 1)
        callbacks.append(learning_rate_decay)

    if save_best == True:
        checkpoint = K.callbacks.ModelCheckpoint(
                filepath = filepath,
                monitor = "val_loss",
                save_best_only = save_best)
        callbacks.append(checkpoint)

    return network.fit(
        x = data,
        y = labels,
        batch_size = batch_size,
        epochs = epochs,
        verbose = verbose,
        shuffle = shuffle,
        validation_data = validation_data,
        callbacks = callbacks)

