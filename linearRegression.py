import numpy as np
import pylab as pl

data = np.loadtxt('/Users/Zohair/Library/Mobile Documents/com~apple~CloudDocs/Education/Machine Learning/machine-learning-ex1/ex1/ex1data1.txt', delimiter=",")
pl.scatter(data[:, 0], data[:, 1], marker='o', c='b')
pl.title('Profits distribution')
pl.xlabel('Population of City in 10,000s')
pl.ylabel('Profit in $10,000s')
pl.show()



(m, n) = np.shape(data) # m is number of examples, n is number of features

temp = np.ones((m, 1)) 
data = np.hstack((temp, data)) # Add column of ones

data = np.random.permutation(data) # Shuffle data

# (trainX, testX) = np.vsplit(data, (400,)) # Split Into Training & Test Set

(X, y) = np.hsplit(data, (n,)) # Split into X and Y values
# (testX, testY) = np.hsplit(testX, (14,)) #Split into X and Y values

(m, n) = np.shape(X)
theta = np.zeros((n,1)) # Initialise Theta Vector

def computeCost(X, y, theta):
    
    # Returns Cost Function J
    
    (m, n) = np.shape(X)
    prediction = np.dot(X,theta)
    
    squareError = np.square(prediction - y)
    J = 1/(2 * m) * np.sum(squareError)
    
    return J

def gradientDescent(X, y, theta, alpha, numIterations):
    
    # Returns Minimised Theta Vector
    
    i = 0
    while i < numIterations:
        
        prediction = np.dot(X, theta)
        errorVector = prediction - y
        thetaChange = alpha * (1/m) * (np.dot(np.transpose(X), errorVector)) # alpha * 1/m * (X' * errorVector)
        theta = theta - thetaChange
        i = i + 1
        
    return theta   

# Alternative Gradient Descent Function

# def gradientDescent(x, y, theta, alpha, numIterations):
#     xTrans = x.transpose()
#     for i in range(0, numIterations):
#         hypothesis = np.dot(x, theta)
#         loss = hypothesis - y
#         # avg cost per example (the 2 in 2*m doesn't really matter here.
#         # But to be consistent with the gradient, I include it)
#         cost = np.sum(loss ** 2) / (2 * m)
#         print("Iteration %d | Cost: %f" % (i, cost))
#         # avg gradient per example
#         gradient = np.dot(xTrans, loss) / m
#         # update
#         theta = theta - alpha * gradient
#     return theta

print ("Initial Cost...")
print (computeCost(X, y, theta))

print ("\n\nRunning Gradient Descent...\n")
alpha = 0.01
numIterations = 1500
print ("Initial Theta...")
print (theta)
print ("\n")
thetaMinimised = gradientDescent(X, y, theta, alpha, numIterations)
print ("Theta Obtained Is...")
print (thetaMinimised)
print ("\n")

# print ("Running Theta Obtained On Test Set...\n")
testCase = np.array([1, 7])
print ("House Price For Given Test Case...\n")
print (10000 * np.dot(testCase, thetaMinimised))


