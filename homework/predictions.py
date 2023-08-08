import numpy as np


def do_predictions(X, t, weights):
    examples, _ = X.shape
    pred = np.dot(X, weights)

    error = pred - t
    cost = np.sum(error ** 2) / (2 * examples)
    print(f'The cost is {cost}')