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

def init(data):
    """ Initialize data sets into workable matrixes."""
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

#This holds the 
#Error_k = []

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

def gradiendDecent(learning_rate, n_iterations, X, Y, e = False):
    W = init_weights(X.shape[1])
    #We need to itterate over m times, to train every weight
    E_ce = [] #so we can test different functions.
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
            E_ce.append(Eck_k)
        #Convergence check
        #if(error_N(W,X,N)):
        #    print("Converge")
        #    return W
            #learning_rate *= 0.5 #half learning rate.
    if(e):
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

#Initalise weights to all zeros
#W = init_weights(X.shape[1])

learning_rate = 0.1



#Function that we want to plot(linear function) with our trained weights.
def n(x):
    return predict_function(W*x.transform())

def p(x):
    return predict_function(W*x.transform())

#Gather values to plot from test data
#X_plot, Y_plot = init_plot(testData_t3)



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

#print(error_N(W,X,Y))

def plot_error(Error_k):
    """ plot the error of the classification after 1000 iterations"""
    #Error_k array with two sets of Emc for test and training set.
    x = np.arange(0, 1000, 1)
    #plot the data
    pyplot.plot(x, Error_k[0],'r--', x, Error_k[1], 'g--')
    pyplot.legend(('Training set', 'Test set'),loc="upper right")
    pyplot.ylim([0,1]) 
    pyplot.show()

def decision_boundry(W, X):
    #points where W^t*X = 0
    boundry = W.transpose()*X.transpose()
    #print(X.shape)
    #print(W.shape)
    print(boundry.shape)
    return boundry.tolist()

def b(x):
    return -np.squeeze(np.asarray(W[0]))/np.squeeze(np.asarray(W[2])) - (np.squeeze(np.asarray(W[1])))/(np.squeeze(np.asarray(W[2])))*x

def b(x,W):
    return -np.squeeze(np.asarray(W[0]))/np.squeeze(np.asarray(W[2])) - (np.squeeze(np.asarray(W[1])))/(np.squeeze(np.asarray(W[2])))*x   

def plot_data(data,W):
    """ Plot data as coloured points according to acctual value"""
    numberOfParameters = data[0].size   
    for example in data:
        """ Check y value of each example and plot accordingly """
        if(example[numberOfParameters-1] == 1):
            pyplot.plot(example[0],example[1], "go")
        elif(example[numberOfParameters-1] == 0):
            pyplot.plot(example[0],example[1], "ro")
    x = np.arange(0,1.1,0.1) #x values to plot the decision boarder.
    pyplot.plot(x,b(x,W),'k--')
    pyplot.xlabel("x1")
    pyplot.ylabel("x2")
    pyplot.ylim([0,1])
    pyplot.xlim([0,1])
    pyplot.show()

def plot_b(data,W):
    """ Plot data """
    numberOfParameters = data[0].size   
    """for example in data:
        if(example[numberOfParameters-1] == 1):
            pyplot.plot(example[0],example[1], "go")
        elif(example[numberOfParameters-1] == 0):
            pyplot.plot(example[0],example[1], "ro")
    """
    x = np.arange(0,1.1,0.1)
    pyplot.plot(x,b(x,W),'k-')
    #pyplot.xlabel("x1")
    #pyplot.ylabel("x2")
    #pyplot.ylim([0,1])
    #pyplot.xlim([0,1])
    pyplot.show()
    

#compute W on the training set
#W = gradiendDecent(learning_rate, 1000, X, Y,False)

#Print error function with each itteration, but very slow cause we need to run 3 times to compute w, and error plots.
"""
Error_k = gradiendDecent(learning_rate, 1000, X, Y, True), gradiendDecent(learning_rate,1000, P, N, True)
plot_error(Error_k)
"""
#Use plot_data to show boundry line on the data
#plot_data(trainData,W)
#plot_data(testData,W)

"""Task 2.2.2"""
#read csv files
trainData_2 = np.loadtxt(open("../classification/cl_train_2.csv", "rb"), delimiter=",")
testData_2 = np.loadtxt(open("../classification/cl_test_2.csv", "rb"), delimiter=",")

#X,Y = init(trainData_2)
#X2,Y2 = init(testData_2)

#W = gradiendDecent(0.01, 1000,X, Y)


#error_value_test = error_N(W,X2,Y2)
#print(error_value_test)

#plot_data(testData_2,W)

#Error_k = gradiendDecent(0.01, 1000, X, Y, True), gradiendDecent(0.01, 1000, X2, Y2, True)
#plot_error(Error_k)

#plot_data(trainData_2, W)

""" Upscaling the h function to actually solve the function"""

def b_2p(x,W): #Positiv solution
    """ Return one x_2 value to plot for an x_1 value, W[n]=wn""" 
    A = np.asscalar(W[4])
    B = np.asscalar(W[2])
    C = np.asscalar(W[0]) + np.asscalar(W[1])*x + (np.asscalar(W[3]))*x**2
    return (- B + np.sqrt(B**2 - 4*A*C) ) / (2*A)


def b_2n(x,W): #Negative solution
    """ Return one x_2 value to plot for an x_1 value, W[n]=wn""" 
    """ Since X is an array of all the values, it calculates every point in one swoop, but this gives us the problem of negative C values in the sqrt, which gives an warning. """
    A = np.asscalar(W[4])
    B = np.asscalar(W[2])
    C = np.asscalar(W[0])+np.asscalar(W[1])*x+(np.asscalar(W[3]))*x**2
    return (- B - np.sqrt(B**2 - 4*A*C) ) / (2*A)

def init_updated(data):
    """ Initialize data sets into workable matrixes with extra data points"""
    X = []
    Y = []

    numberOfParameters = data[0].size

    for example in data:
        Y.append([example[numberOfParameters-1]])
        row = [1]
        for i in range(numberOfParameters-1):
            row.append(example[i])
        #Add more variables
        for i in range(numberOfParameters-1):
            row.append(example[i]*example[i])
        X.append(row)
    
    X = np.matrix(X)
    Y = np.matrix(Y)
    return X,Y

X,Y = init_updated(trainData_2)
X2,Y2 = init_updated(testData_2)

#Train on training set, as before
W = gradiendDecent(0.1, 1000,X, Y)

#Error_k = gradiendDecent(0.1, 1000, X, Y, True), gradiendDecent(0.1, 1000, X2, Y2, True)
#plot_error(Error_k)

def plot_b(data, W):
    """ Plot data """
    numberOfParameters = data[0].size
    for example in data:  
        if(example[numberOfParameters-1] == 1):
            pyplot.plot(example[0],example[1], "go")
        elif(example[numberOfParameters-1] == 0):
            pyplot.plot(example[0],example[1], "ro")

    x = np.arange(0,1,0.00001)
    #Plot boundary function
    pyplot.legend(('Negative', 'Positiv'),loc="upper right")
    pyplot.plot(x,b_2p(x,W),'k-', x,b_2n(x,W),'k-')
    pyplot.xlabel("x1")
    pyplot.ylabel("x2")
    pyplot.ylim([0,1])
    pyplot.xlim([0,1])
    pyplot.show()

plot_b(testData_2,W)
#print(b_2n(0,W))
