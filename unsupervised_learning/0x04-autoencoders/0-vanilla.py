#!/usr/bin/env python3

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    encoder_input = keras.Input(shape = (input_dims, ))
    encoder_net = encoder_input
    for units in hidden_layers:
        encoder_net = keras.layers.Dense(units = units,
                                         activation = "relu")(encoder_net)
    latent_layer = keras.layers.Dense(units = latent_dims,
                                      activation = "relu")(encoder_net)
    encoder = keras.Model(inputs = encoder_input, outputs = latent_layer)

    decoder_input = keras.Input(shape = (latent_dims, ))
    decoder_net = decoder_input
    for units in hidden_layers[::-1]:
        decoder_net = keras.layers.Dense(units = units,
                                         activation = "relu")(decoder_net)
    decoder_output = keras.layers.Dense(units = input_dims,
                                        activation = "sigmoid")(decoder_net)
    decoder = keras.Model(inputs = decoder_input, outputs = decoder_output)

    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(inputs = encoder_input, outputs = auto_output)
    
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
