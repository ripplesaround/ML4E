# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/25 13:38
desc: 采用一对一法
'''

# done 单个svm训练
# done 训练全部的，这个要在svm_oago里面写 svm_oago 训练成功，值得注意的是决策便捷的点
# done 超级绘图

import sys
import copy
import data_process
import numpy as np
import matplotlib.pyplot as plt
import cvxopt.solvers
import logging
import kernel
from sklearn import svm
import matplotlib as mpl

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




        # notice 这里输入的数据界定svm训练了几维
        # self.test_x = self.test_x[:, 2:4]
        # self.pred = np.array(self.pred)[:,2:4]

        # self.feature = self.test_x.shape[1]
        # self.support_multipliers = []
        # self.support_vectors = []
        # self.support_vector_labels = []

    def train_oago(self):
        # test1 class1的数据为-1，class2的数据为+1
        # self.feature = self.test_x.shape[1]
        self.test_x = []
        self.test_x.extend(self.train_data[1])
        self.test_x.extend(self.train_data[2])
        label = np.ones((40, 1))
        self.test_y = []
        self.test_y.extend(label - 2)
        self.test_y.extend(label)
        self.test_x = np.array(self.test_x)
        self.test_y = np.ravel(np.array(self.test_y))
        self.pred = []
        self.pred.extend(self.test_data[1])
        self.pred.extend(self.test_data[2])
        self.pred = np.array(self.pred)
        test1 = svm_train(self.test_x, self.test_y,kernel=kernel.Kernel.radial_basis(1/self.test_x.shape[1]),c = 1)
        test1.train()
        print(test1.predict(self.pred))
        test1.plot(self.pred)

        # test2 class0的数据为-1，class1的数据为+1
        self.test_x = []
        self.test_x.extend(self.train_data[0])
        self.test_x.extend(self.train_data[1])
        label = np.ones((40, 1))
        self.test_y = []
        self.test_y.extend(label - 2)
        self.test_y.extend(label)
        self.test_x = np.array(self.test_x)
        self.test_y = np.ravel(np.array(self.test_y))
        self.pred = []
        self.pred.extend(self.test_data[0])
        self.pred.extend(self.test_data[1])
        self.pred = np.array(self.pred)
        test2 = svm_train(self.test_x, self.test_y, kernel=kernel.Kernel.radial_basis(1 / self.test_x.shape[1]), c=1)
        test2.train()
        print(test2.predict(self.pred))
        test2.plot(self.pred)

        # test3 class0的数据为-1，class2的数据为+1
        self.test_x = []
        self.test_x.extend(self.train_data[0])
        self.test_x.extend(self.train_data[2])
        label = np.ones((40, 1))
        self.test_y = []
        self.test_y.extend(label - 2)
        self.test_y.extend(label)
        self.test_x = np.array(self.test_x)
        self.test_y = np.ravel(np.array(self.test_y))
        self.pred = []
        self.pred.extend(self.test_data[0])
        self.pred.extend(self.test_data[2])
        self.pred = np.array(self.pred)
        test3 = svm_train(self.test_x, self.test_y, kernel=kernel.Kernel.radial_basis(1 / self.test_x.shape[1]), c=1)
        test3.train()
        print(test3.predict(self.pred))
        test3.plot(self.pred)

        self.pred = []
        self.pred.extend(self.test_data[0])
        self.pred.extend(self.test_data[1])
        self.pred.extend(self.test_data[2])
        self.pred = np.array(self.pred)
        ans = self.oago_pred(test1,test2,test3,self.pred)
        print(ans)

        # 开始画图
        x1_min, x1_max = self.data[:, 0].min(), self.data[:, 0].max()  # 第0列的范围
        x2_min, x2_max = self.data[:, 1].min(), self.data[:, 1].max()  # 第1列的范围
        x3_min, x3_max = self.data[:, 2].min(), self.data[:, 2].max()  # 第2列的范围
        x4_min, x4_max = self.data[:, 3].min(), self.data[:, 3].max()  # 第3列的范围
        x1, x2 = np.mgrid[x1_min:x1_max:600j, x2_min:x2_max:600j]
        x3, x4 = np.mgrid[x3_min:x3_max:600j, x4_min:x4_max:600j]
        grid_test = np.stack((x1.flat, x2.flat, x3.flat, x4.flat), axis=1)  # 测试点
        grid_hat = self.oago_pred(test1,test2,test3,grid_test)  # 预测分类值
        grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0',"#89CFF0"])
        plt.xlim(x3_min, x3_max)
        plt.ylim(x4_min, x4_max)
        # print(grid_hat)
        plt.pcolormesh(x3, x4, grid_hat, cmap=cm_light)
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        plt.scatter(self.pred[:, 2], self.pred[:, 3], c=ans, edgecolors='k', s=50, cmap=cm_dark, marker='^')
        self.train_data_plot = []
        self.train_data_plot.extend(self.train_data[0])
        self.train_data_plot.extend(self.train_data[1])
        self.train_data_plot.extend(self.train_data[2])
        self.train_data_plot = np.array(self.train_data_plot)
        label = np.ones((40, 1))
        labels  = []
        labels.extend(label - 1)
        labels.extend(label)
        labels.extend(label + 1)
        labels = np.ravel(np.array(labels))
        plt.scatter(self.train_data_plot[:, 2], self.train_data_plot[:, 3], c=labels, edgecolors='k', s=50, cmap=cm_dark)
        plt.show()

        # clf = svm.SVC(gamma='auto')
        # clf.fit(self.test_x, self.test_y)
        # print('------------------------------------\nsklearn')
        # print(clf.support_)
        # print(clf.n_support_)
        # print(clf.predict(self.pred))
        # print('------------------------------------')

    def oago_pred(self,test1,test2,test3,test_data):
        '''
        实现多分类一对一svm的预测功能
        :param test1: class12的svm
        :param test2: class01的svm
        :param test3: class02的svm
        :param test_data:  测试集
        :return: list，包含预测的结果
        '''
        ans = np.zeros((test_data.shape[0], 1))
        flag1 = test1.predict(test_data)
        flag2 = test2.predict(test_data)
        flag3 = test3.predict(test_data)

        # notice 值得注意的是这里，这个判断在绘制决策边界附近的点的时候很难说，因为决策边界中有可能有情况不符合这种点
        # 即一个错，另一个没有的那种情况
        for i,a in enumerate(flag1):
            b = flag2[i]
            c = flag3[i]
            if a==-1 and b==1:
                ans[i] = 1
            if b==-1 and c==-1:
                ans[i] = 0
            if a==1 and c==1:
                ans[i] = 2

        return np.ravel(ans)



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
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        self.support_multipliers = lagrange_multipliers[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        for i,k in enumerate(support_vector_indices):
            if self._c-lagrange_multipliers[i] > MIN_SUPPORT_VECTOR_MULTIPLIER and k == True:
                print(f"计算bias是用的第{i}号向量")
                self._bias = y[i] - sum(lagrange_multipliers*y*self.K[i,:])
                print(self._bias)
                break


    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        self.K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * self.K)
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

    def cal(self,k):
        result = self._bias
        for z_i, x_i, y_i in zip(self.support_multipliers,
                                 self.support_vectors,
                                 self.support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, k)
        return result

    def cal_show(self,y):
        ans = np.zeros((y.shape[0], 1))
        for i,k in enumerate(y):
            ans[i] = np.sign(self.cal(k))
        return ans

    def predict(self, y):
        ans = np.zeros((y.shape[0],1))
        for i,k in enumerate(y):
            ans[i] = np.sign(self.cal(k))
        return np.ravel(ans)


    def plot(self,pred = []):
        '''
        # todo 这个plot函数是不是应该给大类？
        # done 二维绘图，指的是svm是用二维数据训练出来的
        '''
        class1 = np.array(
            [self.support_vectors[i] for i in range(len(self.support_vectors)) if self.support_vector_labels[i] == -1])
        class2 = np.array(
            [self.support_vectors[i]  for i in range(len(self.support_vectors)) if self.support_vector_labels[i] == 1])
        print('开始画图')
        x1_min, x1_max = self.support_vectors[:, 0].min(), self.support_vectors[:, 0].max()  # 第0列的范围
        x2_min, x2_max = self.support_vectors[:, 1].min(), self.support_vectors[:, 1].max()  # 第1列的范围
        x3_min, x3_max = self.support_vectors[:, 2].min(), self.support_vectors[:, 2].max()  # 第2列的范围
        x4_min, x4_max = self.support_vectors[:, 3].min(), self.support_vectors[:, 3].max()  # 第3列的范围
        x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
        x3, x4 = np.mgrid[x3_min:x3_max:200j, x4_min:x4_max:200j]
        grid_test = np.stack((x1.flat, x2.flat,x3.flat,x4.flat), axis=1)  # 测试点
        grid_hat = self.cal_show(grid_test)  # 预测分类值
        grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r'])
        plt.xlim(x3_min, x3_max)
        plt.ylim(x4_min, x4_max)
        # print(grid_hat)
        plt.pcolormesh(x3, x4, grid_hat, cmap=cm_light)
        plt.scatter(self.data[:, 2], self.data[:, 3], c=self.true_labels, edgecolors='k', s=50, cmap=cm_dark)  # 样本
        plt.xlim(x3_min, x3_max)
        plt.ylim(x4_min, x4_max)

        if len(pred)>0:
            ans = self.predict(pred)
            plt.scatter(pred[:, 2], pred[:, 3], c=ans, edgecolors='k', s=50, cmap=cm_dark,marker = '^')
            plt.xlim(x3_min, x3_max)
            plt.ylim(x4_min, x4_max)


        # plt.title("test", fontsize=15)
        # plt.grid()
        plt.show()


        # k1,k2,k3,k4  = np.mgrid[1:3:3j, 1.5:4.5:3j,4:8:3j, 0:3:3j]
        # print(k1,k2,k3,k4)

        # print(f"类别1中的支持向量个数：{len(class1)}")
        # print(f"类别2中的支持向量个数：{len(class2)}")
        # plt.plot(class1[:, 2], class1[:, 3], 'bo', label="class1")
        # plt.plot(class2[:, 2], class2[:, 3], 'ro', label="class2")
        # plt.legend(loc="best")
        # plt.show()
        # plt.plot(class1[:, 0], class1[:, 1], 'bo', label="class1")
        # plt.plot(class2[:, 0], class2[:, 1], 'ro', label="class2")
        # plt.legend(loc="best")
        # plt.show()


