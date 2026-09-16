#!/usr/bin/env python3
""" flips an image horizontally """
import tensorflow as tf


def flip_image(image):
    """ 
    Flips an image left to right
    """
    return tf.image.flip_left_right(image)
