import numpy as np 

# Loading all Data Given
X_seen=np.load('data\AwA_python\X_seen.npy',encoding='bytes',allow_pickle=True)
Xtest=np.load('data\AwA_python\Xtest.npy',encoding='bytes',allow_pickle=True)
Ytest=np.load('data\AwA_python\Ytest.npy',encoding='bytes',allow_pickle=True)
class_attributes_seen=np.load('data\AwA_python\class_attributes_seen.npy',encoding='bytes',allow_pickle=True)
class_attributes_unseen=np.load('data\AwA_python\class_attributes_unseen.npy',encoding='bytes',allow_pickle=True)

# Function for Finding dot product
def dot_product(i):
    Sc_temp_list=[]
    for j in range(0,40):
        temp= np.dot(class_attributes_unseen[i], class_attributes_seen[j])
        Sc_temp_list.append(temp)
    return Sc_temp_list  

# Sc vector of unseen classes
Sc = [dot_product(i) for i in range(0,10)]

# function to normalized Sc vector
def normalized(i):
    return Sc[i]/np.sum(Sc[i])

# Normalized Sc vector of unseen classes
Sc_normalized = [normalized(i) for i in range(10)]

# Calculating Mean of seen classes
mean_of_seen_classes = []

for i in range(X_seen.shape[0]):
    temp_data = X_seen[i]
    calculated_means = np.mean(temp_data, axis=0)
    mean_of_seen_classes.append(calculated_means)


# Calculating Mean of Unseen classes
mean_of_unseen_classes=[]
for i in range(10):
  sum=0
  for j in range(40):
      sum= sum + (Sc_normalized[i][j] * mean_of_seen_classes[j])

  mean_of_unseen_classes.append(sum)

# Finding nearest prototype
def euclidean_distance(i):
    distance_temp = []
    for j in range(0,10):
        distance = np.linalg.norm(Xtest[i] - mean_of_unseen_classes[j])
        distance_temp.append(distance)
    min_dist=np.argmin(distance_temp) +1 
    return min_dist

# YTest predicted output
YTest_predicted = [euclidean_distance(index) for index,val in enumerate(Xtest)]

count = 0
for i,val in enumerate(Ytest):
    if Ytest[i] == YTest_predicted[i]:
        count= count+1

# Finding accuracy
count = count/6180
print("Accuracy =",count*100)
