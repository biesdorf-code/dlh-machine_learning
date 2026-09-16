#!/usr/bin/env python3
""" rotates an image """
import tensorflow as tf


def rotate_image(image):
    """ rotates an image 90 degrees counter-clockwise """
    return tf.image.rot90(image)
