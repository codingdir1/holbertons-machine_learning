#!/usr/bin/env python3

import tensorflow.keras as keras

class VAELossLayer(keras.layers.Layer):
    def call(self, inputs):
        encoder_input, auto_output, mu, log_sigma = inputs
                            
        reconstruction_loss = keras.ops.sum(
            keras.ops.binary_crossentropy(encoder_input, auto_output), axis=-1
        )
        kl_loss = -0.5 * keras.ops.sum(
            1 + 2 * log_sigma - keras.ops.square(mu) - keras.ops.exp(2 * log_sigma), 
            axis=-1
        )
                                                                    
        total_loss = keras.ops.mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return auto_output

def autoencoder(input_dims, hidden_layers, latent_dims):
    encoder_input = keras.Input(shape = (input_dims, ))
    encoder_net = encoder_input
    for units in hidden_layers:
        encoder_net = keras.layers.Dense(units = units,
                                         activation = "relu")(encoder_net)

    mu = keras.layers.Dense(units = latent_dims,
                            activation = None)(encoder_net)
    log_sigma = keras.layers.Dense(units = latent_dims,
                                   activation = None)(encoder_net)
    def sampler(args):
        mu, log_sigma = args
        normal = keras.random.normal(shape = (keras.ops.shape(mu)[0], latent_dims), 
                                     mean = 0, 
                                     stddev = 1.0, 
                                     seed = None)
        return mu + normal * keras.ops.exp(log_sigma)
    
    latent_layer = keras.layers.Lambda(sampler, 
                                       output_shape=(latent_dims,))([mu, log_sigma])
    encoder = keras.Model(inputs = encoder_input, outputs = [mu, log_sigma, latent_layer])

    decoder_input = keras.Input(shape = (latent_dims, ))
    decoder_net = decoder_input
    for units in hidden_layers[::-1]:
        decoder_net = keras.layers.Dense(units = units,
                                         activation = "relu")(decoder_net)
    decoder_output = keras.layers.Dense(units = input_dims,
                                        activation = "sigmoid")(decoder_net)
    decoder = keras.Model(inputs = decoder_input, outputs = decoder_output)

    auto_output = VAELossLayer()([encoder_input, decoder(latent_layer), mu, log_sigma])
    auto = keras.Model(inputs = encoder_input, outputs = auto_output)
    auto.compile(optimizer = "adam")

    return encoder, decoder, auto
