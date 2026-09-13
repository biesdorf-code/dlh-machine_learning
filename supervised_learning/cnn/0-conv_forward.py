#!/usr/bin/env python3
""" performs forward propagation over a convolutional layer """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')

    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1
    Z = np.zeros((m, h_new, w_new, c_new))

    # convolution process
    for i in range(h_new):
        for j in range(w_new):
            x, y = i * sh, j * sw
            window = A_pad[:, x:x + kh, y:y + kw, :, np.newaxis]
            Z[:, i, j, :] = np.sum(window * W, axis=(1, 2, 3))

    return activation(Z + b)
