import numpy as np
import matplotlib.pyplot as plt


x_train = np.genfromtxt('./data/ridgetrain.txt', delimiter=' ', usecols=0)
x_test = np.genfromtxt('./data/ridgetest.txt', delimiter=' ', usecols=0)
y_train = np.genfromtxt('./data/ridgetrain.txt', delimiter='  ', usecols=1)
y_test = np.genfromtxt('./data/ridgetest.txt', delimiter='  ', usecols=1)

L_val = [2, 5, 20, 50, 100]

def landmark(x, y):
    return np.exp(-0.1*np.square(x.reshape((-1,1)) - y.reshape((1,-1))))


for L in L_val:
    rand_val = np.random.choice(x_train, L, replace=False)
    Id = np.eye(L)
    xf_train = landmark(x_train, rand_val)

    Wt_val = np.dot(np.linalg.inv(np.dot(xf_train.T, xf_train) + 0.1*Id), np.dot(xf_train.T, y_train.reshape((-1,1))))

    land_val = landmark(x_test, rand_val)

    y_pred = np.dot(land_val, Wt_val)

    rmse = np.sqrt(np.mean(np.square(y_test.reshape((-1,1)) - y_pred)))
    print ('RMSE for lambda = ' + str(L) + ' is ' + str(rmse))

    plt.figure(L)
    plt.title('L = ' + str(L) + ', rmse = ' + str(rmse))
    plt.plot(x_test, y_test, 'b*')
    plt.plot(x_test, y_pred, 'r*')

plt.show()
