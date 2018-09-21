import numpy as np


def initWeights(num_visible, num_hidden):
    np_rng = np.random.RandomState(1234)

    weights = np.asarray(np_rng.uniform(
        low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
        high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
        size=(num_visible, num_hidden)))

    # Insert weights for the bias units into the first row and first column.
    weights = np.insert(weights, 0, 0, axis=0)
    weights = np.insert(weights, 0, 0, axis=1)

    return weights

