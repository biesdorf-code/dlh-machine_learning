#!/usr/bin/env python3
""" trains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=patience))
        if learning_rate_decay:
            def schedule(epoch):
                """ inverse time decay of the learning rate """
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(K.callbacks.LearningRateScheduler(schedule,
                                                               verbose=1))

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=validation_data,
                       callbacks=callbacks,
                       verbose=verbose,
                       shuffle=shuffle)
