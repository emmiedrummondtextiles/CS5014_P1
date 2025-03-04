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
#give the gradient expression for w

#then implement a gradient descent based algorithm to learn the parameter

class LassoRegression:
    def __init__(self);
        self.lr = lr 




    
    