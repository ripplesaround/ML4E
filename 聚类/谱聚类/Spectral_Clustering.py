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


class mysc:
    def __init__(self, K, data,gamma=1):
        self.K = K
        self.data = data
        self.N = data.shape[0]
        self.data_dim = data.shape[1]
        self.gamma = gamma

    def sim(self):
        W = np.zeros((self.N,self.N))
        D = np.eye(self.N)
        D2 = np.zeros((self.N, self.N))
        for i in range(self.N):
            D[i][i] = 0
            for j in range(self.N):
                if i!=j:
                    W[i][j] = np.exp(-np.linalg.norm(self.data[i] - self.data[j]) / (2 * self.gamma))
                else:
                    W[i][j] = 0
                D[i][i]+=W[i][j]
            D2[i][i] = 1/ (np.sqrt(D[i][i]))
        Lsym =np.dot(np.dot(D2,W),D2) # Lsym = D(-1/2)* W *D(-1/2)
        return Lsym

    def cal(self):
        Lsym = self.sim()
        eigenvalue, featurevector = np.linalg.eig(Lsym)
        # print(np.dot(Lsym,featurevector[:,0]))
        # print(featurevector[:,0])
        # done 两篇论文的结果不一样。取小还是大的特征值
        # done 标准是取小的，只是这里的Lsym不是传统的Lsym，看论文即可，这里的lambda = 1-lambda
        # 取大
        V = featurevector[:,:self.K+3]
        # 取小
        # V = featurevector[:,featurevector.shape[1]-self.K:featurevector.shape[1]]
        # for i in range(self.K):
        U = np.zeros(V.shape)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U[i][j] = V[i][j] / np.linalg.norm(V[i,:])
        return U

    def cluster_plot(self):
        U = self.cal()
        ans = kmeans(U,self.K,self.data)
        ans.cal()
