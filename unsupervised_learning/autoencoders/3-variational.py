#!/usr/bin/env python3
""" creates a variational autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder

    Returns: encoder, decoder, auto
    """
    K = keras.backend

    def sampling(args):
        """ samples z from the mean and log variance """
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(log_var / 2) * epsilon

    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([mu, log_var])
    encoder = keras.Model(inputs, [z, mu, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    z, mu, log_var = encoder(inputs)
    auto = keras.Model(inputs, decoder(z))

    # KL divergence between N(mu, var) and the N(0, 1) prior
    kl = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
    auto.add_loss(K.mean(kl))

    def reconstruction_loss(y_true, y_pred):
        """ binary cross-entropy summed over the input dimensions """
        return keras.losses.binary_crossentropy(y_true, y_pred) * input_dims

    auto.compile(optimizer='adam', loss=reconstruction_loss)

    return encoder, decoder, auto
