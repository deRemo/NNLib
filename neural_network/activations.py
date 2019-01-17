import numpy as np


# Handler function to make easier the choice's logic
def activations(f):
    act = {
        'sigmoid': (sigmoid, d_sigmoid),
        'tanh': (tanh, d_tanh),
        'relu': (relu, d_relu)
    }
    return act[f]


# Implementation of activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    return 1 - (tanh(x) * tanh(x))


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


def d_relu(x):
    if x <= 0:
        return 0
    else:
        return 1