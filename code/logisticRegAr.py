#Assignment 1 Logistic Regression
import numpy as np
#def h(X,W):
#    return W.transpose()*X

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict_function(z):
    """ Predicts an discrete output based on input"""
    if (sigmoid(z) >= 0.5):
        return 1
    return 0

#TODO: Seperate positiv and negativ examples when plotting

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
    
    X = np.array(X) 
    Y = np.array(Y)
    return X,Y

def init_weights(n):
    W = []
    for i in range(n):
        W.append([0])

    return np.array(W)

def cost_function(m, y, x, h):
    cost = 0
    for example in range(m):
        cost += y*np.log2(h*x) + (1-y)*np.log2(1-h(x))
    pass
    #cost = -1/m 

def gradiendDecent(learning_rate,n_iterations, W, X, Y):
    #We need to itterate over m times, to train every weight
    local_learning_rate = learning_rate
    k = 0 #first iteration
    for n in range(n_iterations):
        sum_n = init_weights(X.shape[1]) 
        for i in range(X.shape[0]): #Itterate over every training example.
            #Note the need for transposing X as well, this is because the way X matrix is represented with one example in every row 
            z = W.transpose()@X[i].transpose()
            val = sigmoid(z) - Y[i]
            print(val.shape)
            value = np.asscalar(sigmoid(z) - Y[i])
            print(value)
            sum_n = sum_n + (value)*X[k].transpose()

        #Update the weigth with this sum.
        tempW = W
        W = W - local_learning_rate*sum_n

        #We need to check if our new weights are the same as last time, and change learning rate
        print(W)
    
#Read csv file
trainData = np.loadtxt(open("../classification/cl_test_1.csv", "rb"), delimiter=",")

X,Y = init(trainData) #Initialize data matrixes

#Initalise weights to all zeros
W = init_weights(X.shape[1])

learning_rate = 0.1

gradiendDecent(learning_rate ,1000, W, X, Y)