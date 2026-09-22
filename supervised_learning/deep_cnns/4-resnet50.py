#!/usr/bin/env python3
""" builds the ResNet-50 architecture """
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds the ResNet-50 architecture """
    init = K.initializers.HeNormal(seed=0)
    X = K.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=init)(X)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    stages = [([64, 64, 256], 3, 1), ([128, 128, 512], 4, 2),
              ([256, 256, 1024], 6, 2), ([512, 512, 2048], 3, 2)]
    for filters, blocks, s in stages:
        x = projection_block(x, filters, s)
        for _ in range(blocks - 1):
            x = identity_block(x, filters)

    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=init)(x)

    return K.models.Model(inputs=X, outputs=x)
