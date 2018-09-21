import numpy as np


def sigmoid(x, derivative=False):
    if derivative == True:
        return x * (1 - x)

    return 1.0 / (1 + np.exp(-x))
