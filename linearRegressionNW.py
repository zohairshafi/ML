import numpy as np

data = np.loadtxt('/Users/Zohair/Desktop/stanford_dl_ex-master/ex1/housing.data')


(m, n) = np.shape(data) # m is number of examples, n is number of features

temp = np.ones((m, 1)) 
data = np.hstack((temp, data)) # Add column of ones

data = np.random.permutation(data) # Shuffle data

(trainX, testX) = np.vsplit(data, (400,)) # Split Into Training & Test Set

(trainX, trainY) = np.hsplit(data, (n,)) # Split into X and Y values
(testX, testY) = np.hsplit(testX, (n,)) # Split into X and Y values

(m, n) = np.shape(trainX)
theta = np.random.random((n,1)) # Initialise Theta Vector

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
        thetaChange = alpha * (np.dot(np.transpose(X), errorVector)) # alpha *  (X' * errorVector)
        test = theta - thetaChange
        theta = theta - test       
        i = i + 1
        
    return theta   

print ("Initial Cost...")
print (computeCost(trainX, trainY, theta))

print ("\n\nRunning Gradient Descent...\n")
alpha = 0.01
numIterations = 400
print ("Initial Theta...")
print (theta)
print ("\n")
thetaMinimised = gradientDescent(trainX, trainY, theta, alpha, numIterations)
print ("Theta Obtained Is...")
print (thetaMinimised)
print ("\n")

# print ("Running Theta Obtained On Test Set...\n")
testCase = np.array([1, 0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.9, 4.98])
print ("House Price For Given Test Case...\n")
print (10000 * np.dot(testCase, thetaMinimised))




