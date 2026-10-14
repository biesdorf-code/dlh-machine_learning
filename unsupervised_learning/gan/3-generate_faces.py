#!/usr/bin/env python3
""" convolutional generator and discriminator for 16x16 faces """
from tensorflow import keras


def convolutional_GenDiscr():
    """ builds the convolutional generator and discriminator """

    def get_generator():
        """ latent vector (16) -> greyscale picture (16, 16, 1) """
        inputs = keras.Input(shape=(16,))
        hidden = keras.layers.Dense(2048, activation='tanh')(inputs)
        hidden = keras.layers.Reshape((2, 2, 512))(hidden)

        # each block doubles the side: 2 -> 4 -> 8 -> 16
        for filters in [64, 16, 1]:
            hidden = keras.layers.UpSampling2D()(hidden)
            hidden = keras.layers.Conv2D(filters, (3, 3),
                                         padding='same')(hidden)
            hidden = keras.layers.BatchNormalization()(hidden)
            hidden = keras.layers.Activation('tanh')(hidden)

        return keras.Model(inputs, hidden, name="generator")

    def get_discriminator():
        """ greyscale picture (16, 16, 1) -> score in [-1, 1] """
        inputs = keras.Input(shape=(16, 16, 1))
        hidden = inputs

        # each block halves the side: 16 -> 8 -> 4 -> 2 -> 1
        for filters in [32, 64, 128, 256]:
            hidden = keras.layers.Conv2D(filters, (3, 3),
                                         padding='same')(hidden)
            hidden = keras.layers.MaxPooling2D()(hidden)
            hidden = keras.layers.Activation('tanh')(hidden)

        hidden = keras.layers.Flatten()(hidden)
        outputs = keras.layers.Dense(1, activation='tanh')(hidden)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
