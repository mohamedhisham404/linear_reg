import numpy as np
from numpy.linalg import norm


def gradient_descent_linear_regression(X, t, step_size = 0.1, precision = 0.00001, max_iter = 10000):
    examples, features = X.shape
    iter = 0
    #cur_weights = np.random.rand(features)         
    cur_weights = np.ones(features, dtype=np.float32)
    #cur_weights = np.array([0, 0.6338432,  0.20894728, 0.00150253])

    state_history, cost_history = [], []
    last_weights = cur_weights + 100 * precision    

    def f(weights):
        pred = np.dot(X, weights)
        error = pred - t
        cost = error.T.dot(error) / (2 * examples)  
        return cost

    def f_dervative(weights):
        pred = np.dot(X, cur_weights)       
        error = pred - t
        gradient = X.T @ error / examples   

        return gradient

    
    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()           
        cost = f(cur_weights)
        gradient = f_dervative(cur_weights)

        # print(f'weights: {cur_weights}\n\tcost: {cost} - gradient: {gradient}')

        state_history.append(cur_weights)
        cost_history.append(cost)
        #print(f'state {state_history[-1]} has \n\tcost {cost_history[-1]} - gradient {gradient}')

        cur_weights -= gradient * step_size   # move in opposite direction
        iter += 1

    print(f'Number of iterations ended at {iter} - with cost {cost} - optimal weights {cur_weights}')
    return cur_weights, cost_history


# if __name__ == "__main__":
#     features, target,data = load_data('dataset_200x4_regression.csv')
#     features = np.hstack([np.ones((features.shape[0], 1)), features])
#     cur_weights, cost_history = gradient_descent_linear_regression(features, target)
    
#     weights = normal(features, target)
#     print(weights)
#     do_predictions(features, target, weights)


    
    