#!/usr/bin/python

import numpy as np
import pylab as pl
from math import exp, log
import scipy.optimize as sp

data = np.loadtxt('/Users/Zohair/Library/Mobile Documents/com~apple~CloudDocs/Education/Machine Learning/machine-learning-ex2/ex2/ex2data2.txt', delimiter = ",")

# Sigmoid Function

def sigmoid (x):
    
    result = []
    size = np.shape(x)
    
    try:
        for i in x.flat:
            result.append((1 / (1 + exp(-i))))
        return np.reshape(result, size)
    except:
        return (1 / (1 + exp(-x)))


# Cost Function and Gradient

def computeCost(X, y, theta, lmbda):

    (m, n) = np.shape(X)
    grad = np.zeros(np.shape(theta))
    # errorVector = np.zeros((n, 1))
    
    prediction = np.dot(X, theta)
    hypothesis = sigmoid(prediction)
    
    J = (1/m) * (- (np.dot(np.transpose(y), np.log(hypothesis))) - np.dot(np.transpose(1 - y), np.log(1 - hypothesis) ))
    errorVector = hypothesis - y
    grad = (1/m) * np.dot(np.transpose(X), errorVector)
    
    theta[1] = 0
    regTerm = (lmbda/2 * m) * np.sum(np.square(theta))
    regTermGrad = (lmbda/m) * theta

    return J, grad

def computeGradient(X, y, theta, lmbda):
    
    (m, n) = np.shape(X)

    prediction = np.dot(X, theta)
    hypothesis = sigmoid(prediction)
    errorVector = hypothesis - y
    grad = (1/m) * np.dot(np.transpose(X), errorVector)
    theta[1] = 0
    regTermGrad = (lmbda/m) * theta
    return grad


# Run Logistic Regression

(m, n) = np.shape(data)

# Add Column Of Ones
temp = np.ones((m, 1)) 
data = np.hstack((temp, data))

(m, n) = np.shape(data)

# Initialise Theta
initialTheta = np.zeros((n - 1, 1))

# Get X and y from data
(X, y) = np.hsplit(data, (n - 1,))

# Initial Cost And Gradient
(cost, grad) = computeCost(X, y, initialTheta, 0)

print ("Cost at initial theta (zeros)", cost)
print ("Gradient at initial theta (zeros):")
print (grad)

# Minimise Cost

thetaFinal = sp.fmin_cg(computeCost, initialTheta, computeGradient)






