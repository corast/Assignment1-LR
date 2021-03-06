#Assignment 1 Logistic Regression
import numpy as np
import matplotlib.pyplot as pyplot

def sigmoid(z):
    """ Calculate the sigmoid value """
    return 1/(1+np.exp(-z))

def init(data):
    """ Initialize data sets into workable seperate matrixes as before.
    X = [[1 x1 x2], [1 x1 x2],...],  Y = [y1,y2,...]"""
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
    """ Initialize the pre trained weights, all zeros in our case. W = [[w0],[w1],..,[wn]] """
    W = []
    for i in range(n):
        W.append([0])

    return np.matrix(W)

#This holds the 
#Error_k = []

def error_N(W, X, Y):
    """ Calculate the error for all the examples X per itteration we are given. Sum of error for all functions."""
    #N is an scalar keeping track of how many examples we got in X
    N = X.shape[0]
    sume = 0 # start sum at 0.
    for i in range(N):
        #Itterate
        z = W.transpose()*X[i].transpose()
        sume += Y[i]*np.log(sigmoid(z)) + (1-np.asscalar(Y[i]))*np.log(1-sigmoid(z))
    return ((-1)*np.asscalar(sume))/N

def error(W,X,y):
    """ Calculate error for one example. Only used for debugging """
    z = W.transpose()*X.transpose()
    error = y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z))
    return error

def gradiendDecent(learning_rate, n_iterations, X, Y, e = False):
    """ main function to find weights. """
    #Initalise weights to all zeros
    W = init_weights(X.shape[1])
    #We need to itterate over m times, to train every weight
    E_ce = [] #so we can test different functions.
    k = 0 #first iteration
    for n in range(n_iterations):
        sum_n = init_weights(X.shape[1])
        for i in range(X.shape[0]): #Itterate over every training example.
            #Note the need for transposing X as well, this is because the way X matrix is represented with one example in every row, instead of every colum as in the assignment paper.
            z = W.transpose()*X[i].transpose() # z we pass to sigmoid function.
            sum_n = sum_n + np.asscalar(sigmoid(z) - Y[i])*X[i].transpose()

        #Update the weigth with this sum.
        tempW = W
        W = W - learning_rate*sum_n
        if(e):
            """ If we are only after the error, but not the weights gained. Very slow and inefficient """
            #Calculate error for k-th iteration.
            Eck_k = error_N(W,X,Y)
            E_ce.append(Eck_k)
        #Convergence check
        #if(error_N(W,X,N)):
        #    print("Converge")
        #    return W
            #learning_rate *= 0.5 #half learning rate.
    if(e):
        """ return error for every itteration, istead of weights trained. """
        #If we just want the error to plot, we instead call this function.
        return E_ce
        #Error_k.append(E_ce)
    return W

#Read csv file
trainData = np.loadtxt(open("../classification/cl_train_1.csv", "rb"), delimiter=",")

testData = np.loadtxt(open("../classification/cl_test_1.csv", "rb"), delimiter=",")

#seperate_data
X,Y = init(trainData) #Initialize data matrixes

P,N = init(testData)

learning_rate = 0.1

#last Error value, if needed
#print(error_N(W,X,Y))

def plot_error(Error_k):
    """ plot the error of the classification after 1000 iterations """
    #Error_k array with two sets of Emc for test and training set.
    x = np.arange(0, 1000, 1)
    #plot the data
    pyplot.plot(x, Error_k[0],'r--', x, Error_k[1], 'g--')
    pyplot.legend(('Training set', 'Test set'),loc="upper right") #name the values
    pyplot.ylim([0,1]) #Limit y value, between 0 and 1. x is already limited by np.arrange 
    pyplot.show()

def b(x,W):
    """ function that correspond to linear boundary with our weights """
    return -np.squeeze(np.asarray(W[0]))/np.squeeze(np.asarray(W[2])) - (np.squeeze(np.asarray(W[1])))/(np.squeeze(np.asarray(W[2])))*x   

def plot_data(data,W):
    """ Plot data as coloured points according to acctual value """
    numberOfParameters = data[0].size   
    for example in data:
        """ Check y value of each example and plot colour accordingly """
        if(example[numberOfParameters-1] == 1):
            pyplot.plot(example[0],example[1], "go")
        elif(example[numberOfParameters-1] == 0):
            pyplot.plot(example[0],example[1], "ro")
    x = np.arange(0,1.1,0.1) #x values to plot the decision bondary.
    pyplot.plot(x,b(x,W),'k--')
    pyplot.xlabel("x1")
    pyplot.ylabel("x2")
    pyplot.ylim([0,1])
    pyplot.show()

def plot_b(data,W):
    """ Plot boundary alone, for debugging """
    x = np.arange(0,1.1,0.1)
    pyplot.plot(x,b(x,W),'k-')
    pyplot.ylim([0,1])
    pyplot.show()
    

 #Commented older tasks.
"""
#compute W on the training set with old weights 
W = gradiendDecent(learning_rate, 1000, X, Y,False)
"""

""" #Print error function with each itteration, but very slow cause we need to run 3 times to compute w, and error plots.
Error_k = gradiendDecent(learning_rate, 1000, X, Y, True), gradiendDecent(learning_rate,1000, P, N, True)
plot_error(Error_k)
"""
"""#Use plot_data to show boundry line on the data
#plot_data(trainData,W)
#plot_data(testData,W)
"""

"""Task 2.2.2"""
#read csv files
trainData_2 = np.loadtxt(open("../classification/cl_train_2.csv", "rb"), delimiter=",")
testData_2 = np.loadtxt(open("../classification/cl_test_2.csv", "rb"), delimiter=",")

X,Y = init(trainData_2)
X2,Y2 = init(testData_2)
#Try to train weights with old boundary function 
#W = gradiendDecent(0.01, 1000,X, Y)

#error_value_test = error_N(W,X2,Y2)
#print(error_value_test)
#plot_data(testData_2,W)

#Error_k = gradiendDecent(0.01, 1000, X, Y, True), gradiendDecent(0.01, 1000, X2, Y2, True)

#plot_error(Error_k)
#plot_data(trainData_2, W)

#Final task.
#Upscaling the h function to actually solve the classification problem.

def b_2p(x,W): #Positiv solution
    """ Return one x_2 value to plot for an x_1 value, W[n]=wn
        Since x is an array of all the x1 values, it calculates every point in one swoop, but this gives us the problem of negative C values in the sqrt, which gives an warning. """
    A = np.asscalar(W[4])
    B = np.asscalar(W[2])
    C = np.asscalar(W[0]) + np.asscalar(W[1])*x + (np.asscalar(W[3]))*x**2
    #Return an array with all points we need to plot for x2
    return (- B + np.sqrt(B**2 - 4*A*C) ) / (2*A)


def b_2n(x,W): #Negative solution
    """ Return one x_2 value to plot for an x_1 value, W[n]=wn 
        Since x is an array of all the values x1, it calculates every point in one swoop, but this gives us the problem of negative C values in the sqrt, which gives an warning. """
    A = np.asscalar(W[4])
    B = np.asscalar(W[2])
    C = np.asscalar(W[0])+np.asscalar(W[1])*x+(np.asscalar(W[3]))*x**2
    #Return an array with all points we need to plot for x2
    return (- B - np.sqrt(B**2 - 4*A*C) ) / (2*A)

def init_updated(data):
    """ Initialize data sets into workable matrixes with extra data points """
    X = []
    Y = []

    numberOfParameters = data[0].size

    for example in data:
        Y.append([example[numberOfParameters-1]])
        row = [1]
        for i in range(numberOfParameters-1):
            row.append(example[i])
        #Add more variables for x1 and x2
        for i in range(numberOfParameters-1):
            row.append(example[i]*example[i])
        X.append(row)
    
    X = np.matrix(X)
    Y = np.matrix(Y)
    return X,Y

#""" Final task"""
X,Y = init_updated(trainData_2)
X2,Y2 = init_updated(testData_2)
#Train on training set, as before
W = gradiendDecent(0.1, 1000,X, Y)
print(W)
""" #Plot error function from both data examples
Error_k = gradiendDecent(0.1, 1000, X, Y, True), gradiendDecent(0.1, 1000, X2, Y2, True)
plot_error(Error_k)
"""

def plot_b(data, W):
    """ Plot data with new boundary function """
    #Number of values per row of data in CVS file.
    numberOfParameters = data[0].size
    for example in data:  
        #plot each example with an corresponding colour representing y value
        if(example[numberOfParameters-1] == 1):
            pyplot.plot(example[0],example[1], "go")
        elif(example[numberOfParameters-1] == 0):
            pyplot.plot(example[0],example[1], "ro")

    x = np.arange(0,1,0.00001)
    #Plot boundary function, and plot settings
    pyplot.legend(('Negative', 'Positiv'),loc="upper right")
    pyplot.plot(x,b_2p(x,W),'k-', x,b_2n(x,W),'k-')
    pyplot.xlabel("x1")
    pyplot.ylabel("x2")
    pyplot.ylim([0,1])
    pyplot.show()

#Run final plot function.
plot_b(testData_2,W)