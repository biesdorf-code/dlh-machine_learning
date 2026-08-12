#!/usr/bin/env python3
""" saves and loads a model's weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """ saves a model's weights """
    network.save_weights(filename)


def load_weights(network, filename):
    """ loads a model's weights """
    network.load_weights(filename)
