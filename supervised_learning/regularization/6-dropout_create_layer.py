#!/usr/bin/env python3
""" creates a layer of a neural network using dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """ creates a layer of a neural network using dropout """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=init)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(layer(prev), training=training)
