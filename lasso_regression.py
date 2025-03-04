# if you use jupyter-lab, switch to %matplotlib inline instead
#matplotlib inline
# %matplotlib notebook
#config Completer.use_jedi = False
import sys
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

   # question 1 

dataset1_df = pd.read_csv('./datasets/dataset1.csv', header=0)
dataset1 = np.array(dataset1_df)
d1X, d1Y = dataset1[:, 0:200], dataset1[:, -1]

#clean data set 

dataset1.head()
dataset1 = dataset1.dropna()

#autograd example studres 

def loss(w, X, y):
	y_pred = X @ w
	error = y - y_pred
	return 0.5 / len(y) * np.sum(error ** 2)

gw = grad(loss, 0)(w0, Xtrain, ytrain) # dloss/dw

gX = grad(loss, 1)(w0, Xtrain, ytrain) # dloss/dX
gy = grad(loss, 2)(w0, Xtrain, ytrain) # dloss/dy

# gradient descent 

w0 = np.random.randn(2) # random guess
gradw = evaluate_grad(w0, x_train, y_train) # gradient
lr = 0.1 # learning rate 
while norm(gradw) > epsilon:
	gradw = evaluate_grad(w0, x_train, y_train)
	w0 = w0 - lr * gradw
	...

#then implement a gradient descent based algorithm to learn the parameter



#lasso class 

class Lasso:
    def __init__(self, lr, lamda, iteration):
        self.lr = lr 


class Regression:
    def __init__(self, learning_rate, iteration, regularisation):
        """
        :param learning_rate: A samll value needed for gradient decent, default value id 0.1.
        :param iteration: Number of training iteration, default value is 10,000.
        """
        self.m = None
        self.n = None
        self.w = None
        self.b = None
        self.regularisation = regularisation # will be the l1/l2 regularization class according to the regression model.
        self.lr = learning_rate
        self.it = iteration


def lasso_loss(X, y, beta, lam):
    n = len(y)
    predictions = X @ beta
    residuals = y - predictions
    loss = (1 / (2 * n)) * np.sum(residuals ** 2) + lam * np.sum(np.abs(beta))
    return loss

def lasso_gradient_descent(X, y, beta, lam, learning_rate, iterations):
    n = len(y)
    for _ in range(iterations):
        predictions = X @ beta
        gradient = -(1 / n) * X.T @ (y - predictions)
        beta = beta - learning_rate * gradient
        
        # Apply soft-thresholding for L1 regularization
        beta = np.sign(beta) * np.maximum(0, np.abs(beta) - learning_rate * lam)
        
    return beta

        



param = {
    "lamda" : 0.1,
    "learning_rate" : 0.1,
    "iteration" : 100
}
print("="*100)
linear_reg = RidgeRegression(**param)

# Train the model.
linear_reg.train(X, y) 

# Predict the values.
y_pred = linear_reg.predict(X)

#Root mean square error.
score = r2_score(y, y_pred)
print("The r2_score of the trained model", score)










#lasso regression 

class Regression:
    def __init__(self, learning_rate, iteration, regularization):
        """
        :param learning_rate: A samll value needed for gradient decent, default value id 0.1.
        :param iteration: Number of training iteration, default value is 10,000.
        """
        self.m = None
        self.n = None
        self.w = None
        self.b = None
        self.regularization = regularization # will be the l1/l2 regularization class according to the regression model.
        self.lr = learning_rate
        self.it = iteration

#medium

# Lasso loss function
def lasso_loss(X, y, beta, lam):
    n = len(y)
    predictions = X @ beta
    residuals = y - predictions
    loss = (1 / (2 * n)) * np.sum(residuals ** 2) + lam * np.sum(np.abs(beta))
    return loss# Gradient descent with soft-thresholding

def lasso_gradient_descent(X, y, beta, lam, learning_rate, iterations):
    n = len(y)
    for _ in range(iterations):
        predictions = X @ beta
        gradient = -(1 / n) * X.T @ (y - predictions)
        beta = beta - learning_rate * gradient
        
        # Apply soft-thresholding
        beta = np.sign(beta) * np.maximum(0, np.abs(beta) - learning_rate * lam)
        
    return beta# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 3)
beta_true = np.array([1.5, 0, -2])
y = X @ beta_true + np.random.randn(100) * 0.1# Initialize beta and parameters
beta_init = np.random.randn(3)
lambda_param = 0.1
learning_rate = 0.01
iterations = 1000# Perform Lasso Regression
beta_hat = lasso_gradient_descent(X, y, beta_init, lambda_param, learning_rate, iterations)
print("Estimated coefficients:", beta_hat)


#task 1.2

#plot the full regularisation path of ^w(λ) for a range of penalty parameter λ

#use the plot to tell which features are relevant and what are their cooresponding weights?



