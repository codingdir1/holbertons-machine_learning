#!/usr/bin/env python3

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    
    reg = K.regularizers.l2(lambtha)

    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(units = layers[i],
                                 activation = activations[i],
                                  kernel_regularizer = reg,
                                input_shape = (nx, )))
        else:
            model.add(K.layers.Dropout(rate = 1 - keep_prob))
            model.add(K.layers.Dense(units = layers[i],
                                    activation = activations[i],
                                    kernel_regularizer = reg))
    return model
