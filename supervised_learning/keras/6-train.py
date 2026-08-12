#!/usr/bin/env python3
""" trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    callbacks = []

    if validation_data is not None and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       callbacks=callbacks,
                       verbose=verbose,
                       shuffle=shuffle)
