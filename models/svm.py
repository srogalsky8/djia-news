import sys,os
sys.path.append(os.path.realpath('..'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.multiclass import OneVsOneClassifier
from preprocess import X_tr, y_tr, X_te, y_te

# select 100 random points and find sigma
# the average of the kth nearest neighbor for each point
n = 100
k = 7
sigma = 0
random.seed(10)
rand_indexes = np.array(random.sample(range(0, len(X_tr)), k=n))
rand_X = X_tr[rand_indexes]
rand_y = y_tr[rand_indexes]
for (i, x) in enumerate(rand_X):
    label = rand_y[i]
    subset_x = X_tr[y_tr == label]
    # get dist to kth nearest neighbor in class
    nbrs = NearestNeighbors(n_neighbors=k).fit(subset_x)
    distances, neighbors_list = nbrs.kneighbors([x])
    sigma += distances[0, k-1]
sigma = sigma/n

print('Running OVO Guassian kernel SVM:')
cs = range(-4,6)
Cs = [2**c for c in cs]
svm_errors = np.zeros(len(cs))
for (idx, C) in enumerate(Cs):
    print(f'C=2^{cs[idx]}')
    svm_model = OneVsOneClassifier(svm.SVC(C=C, kernel='rbf', gamma=1/(2*(sigma**2)))).fit(X_tr, y_tr)
    svm_pred = svm_model.predict(X_te)
    svm_errors[idx] = sum(y_te != svm_pred)/len(y_te)

plt.plot(Cs,svm_errors,linestyle='--', marker='o', label='Gaussian SVM')
min_x1 = Cs[np.argmin(svm_errors)]
min_y1 = np.min(svm_errors)
plt.scatter(min_x1, min_y1,c='r', label='minimum', s=80)
print(f'min error of {min_y1} at C={min_x1}')

plt.xlabel('C')
plt.xscale('log', base=2)
plt.ylabel('Error')
plt.title('Error for one-vs-one degree Gaussian SVM')
plt.legend()
plt.show()