#!/usr/bin/env python3
""" performs back propagation over a convolutional layer """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ performs back propagation over a convolutional layer """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant')
    dA_pad = np.zeros(A_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # back propagation process
    for i in range(h_new):
        for j in range(w_new):
            x, y = i * sh, j * sw
            window = A_pad[:, x:x + kh, y:y + kw, :, np.newaxis]
            dz = dZ[:, i, j, :].reshape(m, 1, 1, 1, c_new)
            dA_pad[:, x:x + kh, y:y + kw, :] += np.sum(W * dz, axis=4)
            dW += np.sum(window * dz, axis=0)

    # unpad
    dA_prev = dA_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db
