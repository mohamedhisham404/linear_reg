import argparse
import numpy as np

from load_data import load_data
from linear_verf import *
from visualizations import *
from normal_equ import normal
from predictions import do_predictions
from gradient_descent_linear_regression import *


if __name__ == "__main__":
    print(" 1.linear verification",'\n'
         ,"2. training with all features",'\n')
    choice=int(input("choose what do you want to do: "))

    # step_size=input("Enter learning rate (default: 0.01): ")
    # step_size=float(step_size)
    # precision=float(input("Enter precision (default: 0.0001): "))
    # precision=float(precision)
    # max_iter=int(input("Enter number of epochs to train (default: 1000) "))
    
    print('0. for no processing ','\n'
        '1. for min/max scaling and ','\n'
        '2. for standrizing')
    option= int(input("choose the processing option: "))  

    if(choice==1):
        print('Verify linear line')
        linear_verf()

    df, data, X, t= load_data('dataset_200x4_regression.csv')
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    if(choice==2):
        print('Learn using the 3 features')
        optimal_weights, cost_history = gradient_descent_linear_regression(X, t,
                                    step_size=0.01, precision = 0.0001, max_iter=1000)

        do_predictions(X, t, optimal_weights)
        visualize_iter(cost_history)
        


   