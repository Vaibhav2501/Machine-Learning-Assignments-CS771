import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# Load the data from the provided pickle file
with open('./data/mnist_small.pkl', 'rb') as file:
    data = pickle.load(file)

X = data['X']  # Inputs
Y = data['Y']  # Output Labels

# PCA computation
pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)

#  t-SNE Calculation
tsne = TSNE(n_components=2, random_state=48)
Xtsne = tsne.fit_transform(X)

def projection_data(X2Dim, labels, title):
    plt.figure(figsize=(10,5))
    for i in range(10):
        ind = np.where(labels == i)[0]
        xpt = X2Dim[ind, 0]
        ypt = X2Dim[ind, 1]
        plt.scatter(xpt,ypt,marker='*',s=20,label=f'Digit {i}',alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.show()


projection_data(Xpca,Y,'PCA Projection')
projection_data(Xtsne,Y,'t-SNE Projection')

