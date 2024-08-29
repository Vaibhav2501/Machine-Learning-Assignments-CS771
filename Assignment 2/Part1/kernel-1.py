import numpy as np
import matplotlib.pyplot as plt

xtrain = np.genfromtxt('./data/ridgetrain.txt', delimiter=' ', usecols=0)
xtest = np.genfromtxt('./data/ridgetest.txt', delimiter=' ', usecols=0)
ytrain = np.genfromtxt('./data/ridgetrain.txt', delimiter='  ', usecols=1)
ytest = np.genfromtxt('./data/ridgetest.txt', delimiter='  ', usecols=1)

def kernel_computation(x, y):
   return np.exp(-0.1*np.square(x.reshape((-1,1)) - y.reshape((1,-1))))

lamda_values=[0.1, 1, 10, 100]
K = kernel_computation(xtrain, xtrain)
Id = np.eye(xtrain.shape[0])

for l in lamda_values:
    alpha_val = np.dot(np.linalg.inv(K + l*Id), ytrain)
    K_test = kernel_computation(xtrain, xtest)
    ypred = (np.dot(alpha_val.T, K_test))

    rmse = np.sqrt(np.mean(np.square(ytest - ypred)))
    print('RMSE for lambda = ' + str(l) + ' is ' + str(rmse))

    plt.figure(l)
    plt.title('lambda = ' + str(l) +', rmse = ' + str(rmse) )
    plt.plot(xtest, ytest, 'b*')
    plt.plot(xtest, ypred, 'r*')

plt.show()    