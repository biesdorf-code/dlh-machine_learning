#!/usr/bin/env python3
"""Transformer decoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder for a transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        N: number of blocks in the decoder
        dm: dimensionality of the model
        h: number of heads
        hidden: number of hidden units in the fully connected layer
        target_vocab: size of the target vocabulary
        max_seq_len: maximum sequence length possible
        drop_rate: dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        x: tensor (batch, target_seq_len) input to the decoder
        encoder_output: tensor (batch, input_seq_len, dm) encoder output
        training: boolean to determine if the model is training
        look_ahead_mask: mask applied to the first multi head attention
        padding_mask: mask applied to the second multi head attention
        Returns: tensor (batch, target_seq_len, dm) with the decoder output
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)
        return x
