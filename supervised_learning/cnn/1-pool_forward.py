#!/usr/bin/env python3
""" performs forward propagation over a pooling layer """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs forward propagation over a pooling layer """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1
    A = np.zeros((m, h_new, w_new, c_prev))

    # pooling process
    for i in range(h_new):
        for j in range(w_new):
            x, y = i * sh, j * sw
            window = A_prev[:, x:x + kh, y:y + kw, :]
            if mode == 'max':
                A[:, i, j, :] = np.max(window, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(window, axis=(1, 2))

    return A
