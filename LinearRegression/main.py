#Assignment 1 Linear Regression

import numpy as np

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
    #X matris which is an matrix of all training examples.
    X,Y = init(data)
    
    #print(X.transpose())
    """ Task 1: Implement OLS (Normal equation)  """
    W = (((X.transpose()*X).I)*X.transpose())*Y
    return W
#Task 1.
trainData = np.loadtxt(open("../data/regression/test_2d_reg_data.csv", "rb"), delimiter=",")
#Load all the data into an matrix, for easy use later. Optimaly we would push to an matrix

#Task 2, train and test.
testData = np.loadtxt(open("../data/regression/train_2d_reg_data.csv", "rb"), delimiter=",")

def test(weights, data):
    error = 0

    for test in data:
        pass 

#mathlibplot