#!/usr/bin/env python3
""" pca color augmentation """
import tensorflow as tf


def pca_color(image, alphas):
    """ pca color augmentation like in the alexnet paper """
    img = tf.cast(image, tf.float32) / 255.0  # same dog, unchanged
    flat = tf.reshape(img, [-1, 3])  # one long strip of pixels
    flat = flat - tf.reduce_mean(flat, axis=0)  # grayish, dark, odd tints
    n = tf.cast(tf.shape(flat)[0] - 1, tf.float32)  # just a number
    cov = tf.matmul(flat, flat, transpose_a=True) / n
    eig_vals, eig_vecs = tf.linalg.eigh(cov)
    scaled = tf.cast(alphas, tf.float32) * eig_vals
    delta = tf.linalg.matvec(eig_vecs, scaled)
    out = tf.clip_by_value(img + delta, 0.0, 1.0)
    return tf.cast(out * 255.0, tf.uint8)
