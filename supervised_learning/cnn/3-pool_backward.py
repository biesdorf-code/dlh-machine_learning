#!/usr/bin/env python3
""" performs back propagation over a pooling layer """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling layer """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)

    # back propagation process
    for i in range(h_new):
        for j in range(w_new):
            x, y = i * sh, j * sw
            da = dA[:, i, j, :].reshape(m, 1, 1, c)
            if mode == 'max':
                window = A_prev[:, x:x + kh, y:y + kw, :]
                mask = window == np.max(window, axis=(1, 2), keepdims=True)
                dA_prev[:, x:x + kh, y:y + kw, :] += mask * da
            else:
                dA_prev[:, x:x + kh, y:y + kw, :] += da / (kh * kw)

    return dA_prev
