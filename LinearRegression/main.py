#Assignment 1 Linear Regression

import numpy as np
import matplotlib.pyplot as pyplot

"""
A = np.matrix([[1, 0.2],[1, 2.4],[1, 2.2],[1, 1.1],[1, 0.6]])
B = np.matrix([[1],[2],[3],[4],[5]])
print(A.transpose())
print(A.transpose()*A)
print(A.transpose()*B)
C = A.transpose()
"""
def init(data):
    X = []
    Y = []

    numberOfParameters = data[0].size

    for example in data:
        Y.append([example[numberOfParameters-1]])
        row = [1]
        for i in range(numberOfParameters-1):
            row.append(example[i])
        X.append(row)
    
    X = np.matrix(X)
    Y = np.matrix(Y)
    return X,Y

def train(data):
    """ Train Linear Regression algorithm with normal equation on input data"""
    #X matrix is an matrix of all training examples.
    X,Y = init(data)
    
    #print(X.transpose())
    """ Task 1: Implement OLS (Normal equation)  """
    W = (((X.transpose()*X).I)*X.transpose())*Y
    return W

#Task 1.
trainData = np.loadtxt(open("../regression/train_2d_reg_data.csv", "rb"), delimiter=",")
#Load all the data into an matrix, for easy use later. Optimaly we would push to an matrix

#Task 2, train and test.
testData = np.loadtxt(open("../regression/test_2d_reg_data.csv", "rb"), delimiter=",")

def test(W, data):
    X,Y = init(data)

    #Equation 6 
    Emse = (X*W - Y).transpose()*(X*W-Y)

    return Emse

#First we train and calculate weights
Weights = train(testData)
Error =  test(Weights, testData)
#np.squeeze(np.asarray(Weights[0])
#Show weights and error after testing
print("w_0 =  {0:.4f}, w_1 = {1:.4f}, w_2 = {2:.4f}, Model error: E_mse(w) = {3:4f}".format(
    np.squeeze(np.asarray(Weights[0])), 
    np.squeeze(np.asarray(Weights[1])), 
    np.squeeze(np.asarray(Weights[2])), 
    np.squeeze(np.asarray(Error))))

#Task 3, plot line after training and plot values

tast3_trainingData = trainData = np.loadtxt(open("../regression/train_1d_reg_data.csv", "rb"), delimiter=",")

W_t3 = train(tast3_trainingData)

def h(x):
    return np.squeeze(np.asarray(W_t3[0])) + (np.squeeze(np.asarray(W_t3[1])))*x
#mathlibplot

#import mathlibplot.pyplot as plot


x = np.arange(0.0, 2.0, 0.1)

pyplot.plot(x, h(x),'r--')
pyplot.show()