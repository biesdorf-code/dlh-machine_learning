#!/usr/bin/env python3
"""RNN decoder for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        vocab: size of the output vocabulary
        embedding: dimensionality of the embedding vector
        units: number of hidden units in the RNN cell
        batch: batch size
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        x: tensor (batch, 1) previous word as a target vocabulary index
        s_prev: tensor (batch, units) previous decoder hidden state
        hidden_states: tensor (batch, input_seq_len, units) encoder outputs
        Returns: y (batch, vocab) output word scores,
                 s (batch, units) new decoder hidden state
        """
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(x)
        y = self.F(tf.reshape(outputs, (-1, outputs.shape[2])))
        return y, s
