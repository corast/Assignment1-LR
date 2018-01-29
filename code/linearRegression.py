#Assignment 1 Linear Regression

import numpy as np
import matplotlib.pyplot as pyplot

def init(data):
    """ Input is the data array, with one example per row, which we split up to seperate X and Y matrixes. And return as numpy matrix objects, for easy matrix calculations"""
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
    #X matrix is an matrix of all training examples, Y corresponding correct answer.
    X,Y = init(data)
    
    """ Task 1: Implement OLS (Normal equation)  """
    W = (((X.transpose()*X).I)*X.transpose())*Y
    return W

#Task 1.
trainData = np.loadtxt(open("../regression/train_2d_reg_data.csv", "rb"), delimiter=",")
#Load all the data into an matrix, for easy use later. Optimaly we would push to an matrix in the same step.

#Task 2, train and test.
testData = np.loadtxt(open("../regression/test_2d_reg_data.csv", "rb"), delimiter=",")

def test(W, data):
    """ Takes the trained weights W, and data to test this on. Return the mean squared error """
    X,Y = init(data)

    #Equation 6 
    Emse = ((X*W - Y).transpose())*(X*W-Y)

    return Emse

#First we train and calculate weights
Weights = train(trainData)


#Print the trained weights.
print(np.asarray(Weights))

#Show weights and error after testing
print("w_0 =  {0}, w_1 = {1}, w_2 = {2}".format(
    np.squeeze(np.asarray(Weights[0])), 
    np.squeeze(np.asarray(Weights[1])), 
    np.squeeze(np.asarray(Weights[2]))))

#Calculate Error for the testing data and training data
Error = test(Weights, testData)
print("Error: testData = ",end = "")
print(np.asscalar(Error))
Error_test = test(Weights, trainData)
print("Error: trainData =  ",end = "")
print(np.asscalar(Error_test))

#Task 3, plot line after training and plot values
""" Task 3, plot boundary line after traing with data """

def init_plot(data):
    """ Split test examples into X and Y array, for easy plotting later. Could have plottet this right away here too"""
    X = []
    Y = []

    for example in data:
        Y.append(example[1])
        X.append(example[0])
    return X,Y

"""Load the test and training data required for task 3 """ 
trainingData_t3 = np.loadtxt(open("../regression/train_1d_reg_data.csv", "rb"), delimiter=",")

testData_t3 = np.loadtxt(open("../regression/test_1d_reg_data.csv", "rb"), delimiter=",")

#Calculat the weights on the trainingdata.
W_t3 = train(trainingData_t3)

#Function that we want to plot(linear function) with our trained weights.
def h(x):
    return np.squeeze(np.asarray(W_t3[0])) + (np.squeeze(np.asarray(W_t3[1])))*x

#Gather values to plot from test data
X_plot, Y_plot = init_plot(testData_t3)

#Plot function from 0 to 1.2 with interval 0.1
x = np.arange(0.0, 1.2, 0.1)

#plot the data
pyplot.title("Hypotesis function and test_data")
pyplot.plot(x, h(x),'r--', X_plot, Y_plot, 'bo' )
pyplot.show()