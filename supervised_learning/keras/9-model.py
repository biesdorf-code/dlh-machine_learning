#!/usr/bin/env python3
""" saves and loads an entire model """
import tensorflow.keras as K


def save_model(network, filename):
    """ saves an entire model """
    network.save(filename)


def load_model(filename):
    """ loads an entire model """
    return K.models.load_model(filename)
