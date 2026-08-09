#!/usr/bin/env python3
""" defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """ deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize the network with nx features and a list of layers."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev))
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

    @property
    def L(self):
        """The number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding all intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding all weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network."""
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A = self.__cache["A{}".format(i)]
            Z = np.matmul(W, A) + b
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the network."""
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)
            self.__weights["W{}".format(i)] = W - alpha * dW
            self.__weights["b{}".format(i)] = (
                self.__weights["b{}".format(i)] - alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the deep neural network over a number of iterations."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
        return self.evaluate(X, Y)
