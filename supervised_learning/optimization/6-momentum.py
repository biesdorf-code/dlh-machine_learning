#!/usr/bin/env python3
""" sets up gradient descent with momentum in TensorFlow """
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """ sets up gradient descent with momentum in TensorFlow """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
