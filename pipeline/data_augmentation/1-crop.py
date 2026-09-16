#!/usr/bin/env python3
""" random crop of an image """
import tensorflow as tf


def crop_image(image, size):
    """ randomly crops an image to size """
    return tf.image.random_crop(image, size)
