#!/usr/bin/env python3

import tensorflow.keras as keras

def autoencoder(input_dims, filters, latent_dims):
    encoder_input = keras.Input(shape = input_dims)
    encoder_net = encoder_input
    for n_filter in filters:
        encoder_net = keras.layers.Conv2D(
                filters = n_filter,
                kernel_size = (3, 3),
                padding = "same",
                activation = "relu")(encoder_net)
        encoder_net = keras.layers.MaxPooling2D(
                pool_size = (2, 2),
                padding = "same")(encoder_net)
    encoder = keras.Model(inputs = encoder_input, outputs = encoder_net)

    decoder_input = keras.layers.Input(shape = latent_dims)
    decoder_net = decoder_input
    for i in range(len(filters) - 1, -1, -1):
        if i == 0:
            padding = "valid"
        else:
            padding = "same"

        decoder_net = keras.layers.Conv2D(
                filters = filters[i],
                kernel_size = (3, 3),
                padding = padding,
                activation = "relu")(decoder_net)
        decoder_net = keras.layers.UpSampling2D(
                size = (2, 2))(decoder_net)
    decoder_output = keras.layers.Conv2D(
            filters = input_dims[2],
            kernel_size = (3, 3),
            padding = "same",
            activation = "sigmoid")(decoder_net)
    decoder = keras.Model(inputs = decoder_input, outputs = decoder_output)

    auto = keras.Model(inputs = encoder_input, outputs = decoder(encoder(encoder_input)))
    auto.compile(loss = "binary_crossentropy", optimizer = "adam")
    
    return encoder, decoder, auto
