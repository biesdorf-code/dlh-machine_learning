#!/usr/bin/env python3
"""Self attention (Bahdanau) for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation"""

    def __init__(self, units):
        """
        Class constructor

        units: number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev: tensor (batch, units) previous decoder hidden state
        hidden_states: tensor (batch, input_seq_len, units) encoder outputs
        Returns: context (batch, units),
                 weights (batch, input_seq_len, 1)
        """
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
