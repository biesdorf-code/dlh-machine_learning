#!/usr/bin/env python3
""" builds a projection block (ResNet) """
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block (ResNet) """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    x = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='same',
                        kernel_initializer=init)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=init)(x)
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s), padding='same',
                               kernel_initializer=init)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    x = K.layers.Add()([x, shortcut])
    return K.layers.Activation('relu')(x)
