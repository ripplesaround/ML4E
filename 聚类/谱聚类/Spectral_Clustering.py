# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/17 21:48
desc:
'''

import sys
import copy
sys.path.append("../k-means")
sys.path.append("../")
import data_process
import numpy as np
import matplotlib.pyplot as plt
from k_means import kmeans
from sklearn.cluster import SpectralClustering




# 采用系统预制的函数
#

def sim(data):
    W = np.zeros(N,N)
    return W

W = sim(data)

print(data)


class sk_sc:
    def __init__(self, K, data,gamma=1):
        self.K = K
        self.data = data
        self.N = data.shape[0]
        self.data_dim = data.shape[1]
        self.gamma = gamma

    def cal_plot(self):
        y_pred = SpectralClustering(n_clusters=self.K, gamma=self.gamma).fit_predict(self.data)
        class1 = np.array([self.data[i] for i in range(self.N) if y_pred[i] == 0])
        class2 = np.array([self.data[i] for i in range(self.N) if y_pred[i] == 1])
        class3 = np.array([self.data[i] for i in range(self.N) if y_pred[i] == 2])
        plt.plot(class1[:, 2], class1[:, 3], 'co', label="class1")
        plt.plot(class2[:, 2], class2[:, 3], 'yo', label="class2")
        plt.plot(class3[:, 2], class3[:, 3], 'go', label="class3")
        plt.legend(loc="best")
        plt.title("Spectral Clustering dim 2/3")
        plt.show()
        plt.plot(class1[:, 0], class1[:, 1], 'co', label="class1")
        plt.plot(class2[:, 0], class2[:, 1], 'yo', label="class2")
        plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
        plt.legend(loc="best")
        plt.title("Spectral Clustering dim 0/1")
        plt.show()
        print("class1", " ", len(class1))
        print("class2", " ", len(class2))
        print("class3", " ", len(class3))






# x = kmeans(path,3)