import numpy as np
from matplotlib import pyplot as plt
from gradient_descent_linear_regression import *
from visualizations import *


def linear_verf():
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    x_vis = X
    t = 5 + X
    X = X.reshape((-1, 1))
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    print('Verify linear line')

    optimal_weights, cost_history = gradient_descent_linear_regression(X, t,
                                step_size=0.1, precision = 0.00001, max_iter=1000)

    # Number of iterations ended at 695 - with cost 4.4770104998508613e-08 - optimal weights [4.99957534 1.00078798]

    pred = np.dot(X, optimal_weights)

    visualize_p1(x_vis,t,pred)