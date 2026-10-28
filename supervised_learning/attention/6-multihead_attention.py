#!/usr/bin/env python3
"""Multi head attention layer"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """
        Class constructor

        dm: dimensionality of the model
        h: number of heads (dm is divisible by h)
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """
        Splits the last dimension into (h, depth) and transposes
        the result to (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Q: tensor (batch, seq_len_q, dk) input to generate the query matrix
        K: tensor (batch, seq_len_v, dk) input to generate the key matrix
        V: tensor (batch, seq_len_v, dv) input to generate the value matrix
        mask: always None
        Returns: output (..., seq_len_q, dm),
                 weights (..., h, seq_len_q, seq_len_v)
        """
        batch = tf.shape(Q)[0]
        q = self.split_heads(self.Wq(Q), batch)
        k = self.split_heads(self.Wk(K), batch)
        v = self.split_heads(self.Wv(V), batch)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch, -1, self.dm))
        output = self.linear(concat)
        return output, weights
