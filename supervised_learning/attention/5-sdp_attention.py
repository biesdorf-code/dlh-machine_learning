#!/usr/bin/env python3
"""Scaled dot product attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention

    Q: tensor (..., seq_len_q, dk) containing the query matrix
    K: tensor (..., seq_len_v, dk) containing the key matrix
    V: tensor (..., seq_len_v, dv) containing the value matrix
    mask: tensor broadcastable to (..., seq_len_q, seq_len_v), or None
    Returns: output (..., seq_len_q, dv), weights (..., seq_len_q, seq_len_v)
    """
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)
    if mask is not None:
        scaled += mask * -1e9
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
