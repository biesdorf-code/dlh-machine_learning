#!/usr/bin/env python3
"""first stack of bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ plots stack of bars, with semantic color choice"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    width = 0.50

    bottom = np.zeros(3)
    for i in range(4):
        plt.bar(people, fruit[i], width=width, color=colors[i],
                label=labels[i], bottom=bottom)
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
