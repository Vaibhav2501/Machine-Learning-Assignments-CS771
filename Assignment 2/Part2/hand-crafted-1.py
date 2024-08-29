import numpy as np
import matplotlib.pyplot as plt 

data = np.genfromtxt('./data/kmeans_data.txt', delimiter='  ')
# print(data)

# Visualize the original data
plt.scatter(data[:, 0], data[:, 1])
plt.title("Original Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

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



funct = np.sum(data**2, axis=1, keepdims=True)
temp_u = funct[:2, :]

for _ in range(10):
    c = predict(funct, temp_u)
    temp_u = means_calculation(funct, c)

cluster_1 = data[c[:, 0] == 0]
cluster_2 = data[c[:, 0] == 1]

# plt.figure(figsize=(5,5))
plt.scatter(cluster_1[:, 0], cluster_1[:, 1], c='r', label='Cluster 1')
plt.scatter(cluster_2[:, 0], cluster_2[:, 1], c='g', label='Cluster 2')
plt.title('Clustering Final Results')
plt.legend()
plt.show()



