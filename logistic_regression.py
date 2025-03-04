#Question 2 (Logistic regression)

import matplotlib.pyplot as plt
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    
from autograd import hessian
import autograd.numpy.linalg as linalg
import matplotlib.pyplot as plt
import pandas as pd



def finite_difference_gradient(f, initial, eps=1e-6):
    initial = np.array(initial, dtype=float)
    n = len(initial)
    output = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        f1 = f(initial + eps * ei)
        f2 = f(initial - eps * ei)
        output[i] = (f1-f2)/(2*eps)
    output = output.reshape(n,1)
    return output

     

# read in dataset1
dataset1_df = pd.read_csv('./datasets/dataset1.csv', header=0)
dataset1 = np.array(dataset1_df)
d1X, d1Y = dataset1[:, 0:200], dataset1[:, -1]





dataset2_df = pd.read_csv('./datasets/dataset2.csv', header=0)
dataset2 = np.array(dataset2_df)
d2X, d2Y = dataset2[:, 0:4], dataset2[:, -1]
# split the data into training and testing 
# the training dataset has the first 1500 observation; 
# in practice, you should randomly shuffle before the split
d2_xtrain, d2_ytrain = d2X[0:1500, :], d2Y[0:1500]
# the testing dataset has the last 500
d2_xtest, d2_ytest = d2X[1500:, :], d2Y[1500:]


def sigmoid(z):
    return 0

def cross_entropy_loss_with_gradient(w, b, X, y):
    ## compute the loss 

	## compute the gradient w.r.t w and b
    
	## return the loss and the required gradients
    return 0

def logistic_regression_train(X, y, lr, tol= 1e-5, maxIters= 2000):
    n, d = X.shape 
    # initialise w0, b0
    # w0 = np.zeros(d)
    # b0 = 0.0
    losses = []
    # loop until converge
    for i in range(maxIters):
        ## Implement gradient descent here
        w0 = w0
        # Check convergence here 
        # if True:
        #    break
    return w0, b0, losses

## run your algorithm and report your findings

#task 2.2



def logistic_regression_reg_train(X, y, lr, lam = 0.01, tol= 1e-5, maxIters= 2000):
    n, d = X.shape 
    # initialise w0, b0
    w0 = np.zeros(d)
    b0 = 0.0
    losses = []
    # loop until converge
    # for i in range(maxIters):
        
    # return w0, b0, losses

## run your algorithm and report your findings




#Task 2.3 Newton's method (extension)

## run Newton's method 

## report your findings

###Task 2.4 Weighted logistic regression (extension)

## gradient expression here

## Implement and run your algorithm and report your findings