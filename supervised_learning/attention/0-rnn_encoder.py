#!/usr/bin/env python3
"""RNN encoder for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes input sequences for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        vocab: size of the input vocabulary
        embedding: dimensionality of the embedding vector
        units: number of hidden units in the RNN cell
        batch: batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to zeros
        Returns: tensor of shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        x: tensor (batch, input_seq_len) of word indices
        initial: tensor (batch, units) with the initial hidden state
        Returns: outputs (batch, input_seq_len, units),
                 hidden (batch, units) last hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
