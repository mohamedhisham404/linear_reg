import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from matplotlib import pyplot as plt


def load_data():
    data = pd.read_csv('dataset_200x4_regression.csv')
    # data= df.to_numpy()
    # scale=MinMaxScaler()
    # data=scale.fit_transform(data)
    
    scale=StandardScaler()
    data=scale.fit_transform(data)
    features = data[:, :3]
    target = data[:, -1]
    return features, target,data


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

def visualize_iter(cost_history):
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        # plt.xlim(0, 100)
        # plt.ylim(0, 0.08)
        plt.grid()
        plt.plot(list(range(len(cost_history))), cost_history, '-r')
        plt.show()    

def visualize_columns(data):
    import seaborn as sns
    sns.pairplot(data, x_vars=['Feat1', 'Feat2', 'Feat3'], y_vars='Target', height=4, aspect=1, kind='scatter')

    plt.show()

if __name__ == "__main__":
    features, target,data = load_data()
    features = np.hstack([np.ones((features.shape[0], 1)), features])
    cur_weights, cost_history = gradient_descent_linear_regression(features, target)
    # visualize_iter(cost_history)
    visualize_columns(data)


    
    