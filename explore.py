import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sklearn as sklearn
import scipy
from clean_data import unique_list, unique_dict, X, y

np.random.seed(10)

def get_euclidean_dist(a,B):
    return np.linalg.norm(B-a,ord=2,axis=1)

def do_knn(X_tr, y_tr, X_te, y_te, k):
    # regular knn for comparison
    errors = np.zeros(k)
    y_pred = np.zeros((k, len(y_te)))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_tr)
    distances, neighbors_list = nbrs.kneighbors(X_te)
    for (index, neighbors) in enumerate(neighbors_list):
        for j in range(k):
            predicted = mode(y_tr[neighbors[0:j+1]])
            y_pred[j, index] = predicted
            if(predicted != y_te[index]):
                errors[j] += 1
                
    # plot errors
    error_rates = errors/len(X_te)
    return (error_rates, y_pred)

def do_nlc(X_tr, y_tr, X_te, y_te, k):
    classes = np.unique(y_tr)
    nlc_errors = np.zeros(k)
    # for each point, create array of classes containing 12 nearest neighbor per class
    neighbors_class = np.zeros((len(X_te), len(classes), k+1, len(X_tr[0])))
    for (j, c) in enumerate(classes):
        class_indices = (y_tr == c)
        class_X = X_tr[class_indices]
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(class_X)
        distances, neighbors_list = nbrs.kneighbors(X_te)
        for (index, neighbors) in enumerate(neighbors_list):
            neighbors_class[index, j] = class_X[neighbors]

    # for each point, get the closest centroid, then check for error
    for (index, point_class) in enumerate(neighbors_class):
        for j in range(k):
            centroids = np.mean(point_class[::,0:j+1], axis=1)
            min_centroid_idx = np.argmin((get_euclidean_dist(X_te[index], centroids)))
            if classes[min_centroid_idx] != y_te[index]:
                nlc_errors[j] += 1

    # plot errors
    nlc_error_rates = nlc_errors/len(X_te)
    return nlc_error_rates

def do_svd(X_tr):
    # 1. center data X_tilde
    x_tr_bar = np.mean(X_tr, axis=0)
    X_tr_tilde = X_tr - x_tr_bar
    # 2. rank k SVD = U_k * Sigma_k * V_k.T
    svd = (np.linalg.svd(X_tr_tilde, full_matrices=False))
    U = svd[0]
    sigma = svd[1]
    V = svd[2]
    return (U, sigma, V)

def do_lda(X_tr, y_tr, X_te):
    classes = np.unique(y_tr)
    S_b = 0
    for (j, c) in enumerate(classes):
        points_in_c = X_tr[y_tr == c]
        class_centroid = np.mean(points_in_c, axis=0)
        m_j_tilde = class_centroid - np.mean(X_tr, axis=0)
        m_j_tilde = np.reshape(m_j_tilde,(1, m_j_tilde.size))
        S_b += len(points_in_c)*np.matmul(m_j_tilde.T, m_j_tilde)

    S_w = 0
    for (j, c) in enumerate(classes):
        points_in_c = X_tr[y_tr == c]
        class_centroid = np.mean(points_in_c, axis=0)
        x_j_tilde = points_in_c - class_centroid
        S_w += np.matmul(x_j_tilde.T, x_j_tilde)

    (e, V) = scipy.linalg.eig(S_b, S_w)
    V_k = V[0:len(classes)-1]
    X_tr_lda = np.matmul(X_tr, V_k.T)
    X_te_lda = np.matmul(X_te, V_k.T)
    return (X_tr_lda, X_te_lda)

k = 12

order = np.random.permutation(len(X))
X_rand = X[order]
y_rand = y[order]
cutoff = len(X_rand) * 4 // 5
X_tr = X_rand[:cutoff]
y_tr = y_rand[:cutoff]
X_te = X_rand[cutoff:]
y_te = y_rand[cutoff:]

# Question 1
def knn_results():
    # PCA 100%
    # (U, sigma, V) = do_svd(X_tr)
    # PC = np.matmul(U, np.diag(sigma))
    # sigma_sq = np.square(sigma)
    # # PCA 95%
    # k1_pca = 0
    # for i in range(len(sigma)):
    #     k1_pca = i+1
    #     if(sum(sigma_sq[0:i+1])/sum(sigma_sq) > 0.95):
    #         break
    # PC_k1 = PC[:, :k1_pca]
    # # transform test data
    # V_k1 = V[:k1_pca]
    # PC_k1_test = np.matmul(X_te - np.mean(X_tr, axis=0), V_k1.T)

    # (pca_k1_error_rates, y_pred) = do_knn(PC_k1, y_tr, PC_k1_test, y_te, k)
    (error_rates, y_pred) = do_knn(X_tr, y_tr, X_te, y_te, k)

    print(len(y_pred))

    plt.plot(np.arange(1,k+1),error_rates,linestyle='--', marker='o', label='knn')
    min_x3 = np.argmin(error_rates)+1
    min_y3 = np.min(error_rates)
    plt.scatter(min_x3, min_y3,c='r', label='minimum error', s=80)
    print(f'knn: Min error of {min_y3} at k={min_x3}')
    print(f'knn: max accuracy of {1-min_y3} at k={min_x3}')
    print(sklearn.metrics.f1_score(y_te, y_pred[min_x3-1], average='macro'))

    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title('Knn errors')
    plt.legend()
    plt.show()

def get_most_common(A, n=3):
    word_freq = np.sum(A, axis=0)
    freq_order = np.argsort(word_freq)[::-1]
    most_common = np.array(unique_list)[freq_order]
    return most_common[0:n]

def plot_distr():
    # get_most_common(X)
    most_common_neg = (get_most_common(X[y==0], 4))
    most_common_pos = (get_most_common(X[y==1], 3))
    most_common_neu = (get_most_common(X[y==2], 3))

    (most_common, idx) = np.unique(np.concatenate([most_common_neg, most_common_pos, most_common_neu]), return_index=True)
    most_common = most_common[idx.argsort()]

    labels = np.unique(y)
    plt_data = np.zeros((len(labels), len(most_common)))
    for (i, label) in enumerate(labels):
        freqs = np.sum(X[y==label], axis=0)
        for (j, word) in enumerate(most_common):
            plt_data[i, j] = freqs[unique_dict[word]]

    print(labels)
    print(most_common)
    print(plt_data)

    plt_X = np.arange(len(most_common))

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    plt.bar(plt_X + 0.00, plt_data[0], color = 'r', width = 0.25)
    plt.bar(plt_X + 0.25, plt_data[1], color = 'g', width = 0.25)
    plt.bar(plt_X + 0.50, plt_data[2], color = 'gray', width = 0.25)
    plt.ylabel('Occurences')
    plt.xlabel('Common words')
    plt.title('Frequencies of most common words across labels')
    plt.xticks(plt_X, most_common)
    plt.legend(labels=['Negative', 'Positive', 'Neutral'])
    plt.show()
    


knn_results()

