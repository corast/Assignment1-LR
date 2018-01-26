#Assignment 1 Logistic Regression
import numpy as np
import matplotlib.pyplot as pyplot
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
def seperate_data(data):
    """ Split test examples into positiv and negativ examples for easy visualization when plotting """
    Xneg = []

    Xpos = []

    numberOfParameters = data[0].size     

    for example in data:
        if(example[1] == 1):
            Xpos.append(example[0])
        else:
            Xneg.append(example[0])
    
    return Xneg,Xpos

def init_plot_data(data):
    """ Split test examples into X and Y array, for easy plotting """
    X = []
    Y = []

    numberOfParameters = data[0].size
    for example in data:
        Y.append(example[1])
        row = [1]
        for i in range(numberOfParameters-1):
            row.append(example[i])
        X.append(row)
    return X,Y 

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

def init_weights(n):
    W = []
    for i in range(n):
        W.append([0])

    return np.matrix(W)

def cost_function(m, y, x, h):
    cost = 0
    for example in range(m):
        cost += y*np.log2(h*x) + (1-y)*np.log2(1-h(x))
    pass
    #cost = -1/m 

Error_k = []

def error_N(W, X, Y):
    N = X.shape[0]
    sume = 0
    for i in range(N):
        #Itterate
        z = W.transpose()*X[i].transpose()
        sume += Y[i]*np.log(sigmoid(z)) + (1-np.asscalar(Y[i]))*np.log(1-sigmoid(z))
    return ((-1)*np.asscalar(sume))/N

def error(W,X,y):
    """ Calculate error per example. """
    z = W.transpose()*X.transpose()
    error = y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z))
    return error

def gradiendDecent(learning_rate, n_iterations, W, X, Y, e = False):
    #We need to itterate over m times, to train every weight
    k = 0 #first iteration
    for n in range(n_iterations):
        sum_n = init_weights(X.shape[1])
        for i in range(X.shape[0]): #Itterate over every training example.
            #Note the need for transposing X as well, this is because the way X matrix is represented with one example in every row 
            #print("{} {} {} {} {} {} {}".format(X[1].shape, W.shape,Y.shape, Y.transpose().shape, X.transpose().shape))
            z = W.transpose()*X[i].transpose()
            sum_n = sum_n + np.asscalar(sigmoid(z) - Y[i])*X[i].transpose()


        #Update the weigth with this sum.
        tempW = W
        W = W - learning_rate*sum_n
        if(e):
            #Calculate error for k-th iteration.
            Eck_k = error_N(W,X,Y)
            Error_k.append(Eck_k)
        #We need to check if our new weights are the same as last time, and change learning rate
        #for e in range(W.shape[0]):
           # if(tempW[0])
        #Convergence check
        #if(error_N(W,X,N)):
        #    print("Converge")
        #    return W
            #learning_rate *= 0.5 #half learning rate.
    return W

#Read csv file
trainData = np.loadtxt(open("../classification/cl_train_1.csv", "rb"), delimiter=",")

testData = np.loadtxt(open("../classification/cl_test_1.csv", "rb"), delimiter=",")
#seperate_data

X,Y = init(trainData) #Initialize data matrixes

P,N = seperate_data(testData)

#Initalise weights to all zeros
W = init_weights(X.shape[1])

learning_rate = 0.1

W = gradiendDecent(learning_rate, 1000, W, X, Y,True)

#Function that we want to plot(linear function) with our trained weights.
def n(x):
    return predict_function(W*x.transform())

def p(x):
    return predict_function(W*x.transform())

#Gather values to plot from test data
#X_plot, Y_plot = init_plot(testData_t3)

#Plot function from 0 to 1.2 with interval 0.1
x = np.arange(0.0, 1.2, 0.1)

def test_data(W):
    X,Y = init_plot_data(testData)

    for example in range(X.shape[0]):
        predict_function(X)

def error_N(W, X, Y):
    N = X.shape[0]
    sume = 0
    for i in range(N):
        #Itterate
        z = W.transpose()*X[i].transpose()
        sume += Y[i]*np.log(sigmoid(z)) + (1-np.asscalar(Y[i]))*np.log(1-sigmoid(z))
    return ((-1)*np.asscalar(sume))/N

def error(W,X,y):
    """ Calculate error per example. """
    z = W.transpose()*X.transpose()
    error = y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z))
    return error

print(error_N(W,X,Y))

x = np.arange(0, 1000, 1)
#plot the data
pyplot.title("decision boundry and test_data")
pyplot.plot(x, Error_k,'r--')
pyplot.show()