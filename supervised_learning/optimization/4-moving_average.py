#!/usr/bin/env python3
""" calculates the weighted moving average of a data set """


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set """
    averages = []
    v = 0

    for i, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        averages.append(v / (1 - beta ** i))

    return averages
