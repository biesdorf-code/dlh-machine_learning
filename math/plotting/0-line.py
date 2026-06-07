#!/usr/bin/env python3
""" draws a line """
import numpy as np
import matplotlib.pyplot as plt

def line():
    """"draws a test redline"""

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()