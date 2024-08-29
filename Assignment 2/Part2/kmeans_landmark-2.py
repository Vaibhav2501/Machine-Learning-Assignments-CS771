import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('./data/kmeans_data.txt', delimiter='  ')
# print(data)

#Visualize the original data
plt.scatter(data[:, 0], data[:, 1])
plt.title("Original Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

def landmark(x, y):
    #  = landmark(x_train, x_train)
    return np.exp(-0.1*np.sum(np.square(x - y.reshape((1,-1))), axis=1)).reshape(-1,1)

def distance(x, u):
    diff = x[:, np.newaxis, :] - u[np.newaxis, :, :]
    dist = np.sum(diff**2, axis=2)
    return dist

def predict(x, u):
    d = distance(x, u)
    c = np.argmin(d, axis=1)
    return c[:, np.newaxis]

def means_calculation(x, c):
    num_clust = len(np.unique(c))
    u = np.zeros((num_clust, x.shape[1]))  
    for k in range(num_clust):
        u[k, :] = np.mean(x[c == k], axis=0) 
    return u

for iter in range(10):
    z = (np.random.randint(0,len(data-1)))
    land_val = landmark(data, data[z,:])
    temp_u = land_val[:2, :]
    c = predict(land_val, temp_u)

    temp_u = means_calculation(land_val, c)
    c = predict(land_val, temp_u)
    cluster_1 = data[c[:, 0] == 0]
    cluster_2 = data[c[:, 0] == 1]
    plt.figure(iter)
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], c='r')
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], c='g')
    plt.plot(data[z,0], data[z,1], 'b*')
    plt.show()
