import numpy as np 

# Load the given data
X_seen = np.load('data/AwA_python/X_seen.npy', encoding='bytes', allow_pickle=True)
Xtest = np.load('data/AwA_python/Xtest.npy', encoding='bytes', allow_pickle=True)
Ytest = np.load('data/AwA_python/Ytest.npy', encoding='bytes', allow_pickle=True)
class_attributes_seen = np.load('data/AwA_python/class_attributes_seen.npy', encoding='bytes', allow_pickle=True)
class_attributes_unseen = np.load('data/AwA_python/class_attributes_unseen.npy', encoding='bytes', allow_pickle=True)

# Define lambda values to test
lambda_given = [0.01, 0.1, 1, 10, 20, 50, 100]

# Calculate the mean of seen classes
mean_of_seen_classes = [np.mean(X_seen[i], axis=0) for i in range(X_seen.shape[0])]

# Calculating Weight vector
def weight_matrix(class_attributes_seen, lambda_regression):
    CAS = class_attributes_seen
    MSC = mean_of_seen_classes
    W = np.linalg.inv(np.transpose(CAS).dot(CAS) + lambda_regression * np.identity(85)).dot(np.transpose(CAS)).dot(MSC)
    return W

# Calculating Mean of Unseen classes
def mean_unseen_class(w):
    mean_of_unseen_classes = [np.dot(class_attributes_unseen[i], w) for i in range(10)]
    return mean_of_unseen_classes

# Calculating accuracy
accuracy_list = []

def accuracy_calculation(mean_of_unseen_classes):
    count = 0
    for i in range(len(Xtest)):
        distance_temp = []
        for j in range(10):
            distance = np.linalg.norm(Xtest[i] - mean_of_unseen_classes[j])
            distance_temp.append(distance)
        min_dist = np.argmin(distance_temp) + 1
        if Ytest[i] == min_dist:
            count += 1
    accuracy = count / len(Xtest) * 100
    return accuracy

# Iterate for all lambda values
for l in lambda_given:
    Weight_vector = weight_matrix(class_attributes_seen, l)
    mean_of_unseen_classes = mean_unseen_class(Weight_vector)
    acc_result = accuracy_calculation(mean_of_unseen_classes)
    accuracy_list.append(acc_result)

# Finding best lambda and accuracy
best_index = np.argmax(accuracy_list) + 1
best_acc_lambda = accuracy_list[best_index-1]
best_lambda_value = lambda_given[best_index-1]
print('Accuracy list for all lambda :- ',accuracy_list)
print('Best accuracy among all:- ',best_acc_lambda)
print('Lambda value for best accuracy:- ',best_lambda_value)

