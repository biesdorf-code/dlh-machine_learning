#!/usr/bin/env python3
""" determines if gradient descent should be stopped early """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if gradient descent should be stopped early """
    count = 0 if opt_cost - cost > threshold else count + 1
    return count >= patience, count
