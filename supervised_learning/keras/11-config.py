#!/usr/bin/env python3
""" saves and loads a model's configuration in json format """
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model's configuration in json format """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """ loads a model with a specific configuration """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
