#!/usr/bin/env python3
""" builds a neural network with the keras functional api """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the keras functional api """
    regularizer = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    return K.Model(inputs=inputs, outputs=x)
