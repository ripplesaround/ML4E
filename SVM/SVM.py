# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/25 13:38
desc: 采用一对一法
'''
import sys
import copy
import data_process
import numpy as np
import matplotlib.pyplot as plt
import cvxopt.solvers
import logging
import kernel
from sklearn import svm

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5
class svm_oago:
    def __init__(self,data,true_labels,kernel=kernel.Kernel.radial_basis(),c = 1,K = 3,train_size=40):
        self.data = data
        self.true_labels = true_labels
        self.cluster = K  # 共有几类
        self.NUM = len(data)
        self.train_data = []
        self.test_data = []
        for i in range(self.cluster):
            temp = []
            temp1 = []
            temp_choice = np.random.choice(a=(self.NUM//self.cluster),size = train_size,replace = False)
            temp_choice += i * (self.NUM//self.cluster)
            for i in range(i*(self.NUM//self.cluster),(i+1)*self.NUM//self.cluster):
                if i in temp_choice:
                    temp.append(self.data[i])
                else:
                    temp1.append(self.data[i])
            self.train_data.append(np.array(temp))
            self.test_data.append(np.array(temp1))
        self.train_size = train_size
        self._kernel = kernel
        self._c = c
        self.test_x =[]
        self.test_x.extend(self.train_data[1])
        self.test_x.extend(self.train_data[2])
        label = np.ones((40,1))
        self.test_y =[]
        self.test_y.extend(label-2)
        self.test_y.extend(label)
        self.test_x = np.array(self.test_x)
        self.test_y = np.ravel(np.array(self.test_y))
        self.pred = []
        self.pred.extend(self.train_data[1])
        self.pred.extend(self.train_data[2])

        self.feature = self.test_x.shape[1]

    def train_oago(self):
        test = svm_train(self.test_x, self.test_y,kernel=kernel.Kernel.radial_basis(1/self.feature),c = 1)
        test.train()
        clf = svm.SVC(gamma='auto')
        clf.fit(self.test_x, self.test_y)
        print('------------------------------------\nsklearn')
        print(clf.support_)
        print(clf.n_support_)
        print(clf.predict(self.pred))
        print('------------------------------------')


class svm_train:
    def __init__(self,data,true_labels,kernel=kernel.Kernel.radial_basis(),c = 1):
        self.data = data
        self.true_labels = true_labels
        self._kernel = kernel
        self._c = c
        self._bias = 0

    def train(self):
        X = self.data
        y = self.true_labels
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        # for i,k in enumerate(support_vector_indices ):
        #     if self._c-lagrange_multipliers[i] < MIN_SUPPORT_VECTOR_MULTIPLIER:
        #         support_vector_indices[i] = False
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        class1 = np.array([support_vectors[i] for i in range(len(support_vectors)) if support_vector_labels[i] == -1])
        class2 = np.array([support_vectors[i] for i in range(len(support_vectors)) if support_vector_labels[i] == 1])
        print(lagrange_multipliers)
        print(support_vector_indices)
        print(len(class1))
        print(len(class2))
        plt.plot(class1[:, 2], class1[:, 3], 'bo', label="class1")
        plt.plot(class2[:, 2], class2[:, 3], 'ro', label="class2")
        plt.legend(loc="best")
        plt.show()
        plt.plot(class1[:, 0], class1[:, 1], 'bo', label="class1")
        plt.plot(class2[:, 0], class2[:, 1], 'ro', label="class2")
        plt.legend(loc="best")
        plt.show()


        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        for i,k in enumerate(support_vector_indices):
            if self._c-lagrange_multipliers[i] > MIN_SUPPORT_VECTOR_MULTIPLIER and k == True:
                print(i)
                break


    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()



