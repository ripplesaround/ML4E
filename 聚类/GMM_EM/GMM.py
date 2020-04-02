# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/11 22:19
desc:
'''
import sys
sys.path.append('../')
import copy

import data_process
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 考虑用k-means迭代后的center来更新结果

K = 3
# 数据集规模
NUM = 150

path = '../iris/iris.data'
data, true_labels= data_process.read_file_list(path)
data = np.array(data)
# data = data_process.scale(data)
# print(data)
print('读入数据完毕')

# 第 k 个模型的高斯分布密度函数
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

# E步
def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]
    # 响应度矩阵
    gamma = np.mat(np.zeros((N, K)))
    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)
    # 计算每个模型对每个样本的响应度
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

# M步
def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]
    #初始化参数值
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)
    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk
        # 更新 cov
        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)
        # 更新 alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha

def init_params(shape, K,max_,min_):
    N, D = shape
    # 采用k-means
    k = [[5.901612903225807, 2.7483870967741932, 4.393548387096775, 1.4338709677419355],
      [6.8500000000000005, 3.0736842105263156, 5.742105263157893, 2.0710526315789473], [5.006, 3.418, 1.464, 0.244]]
    mu = np.array(k)
    mu = scale_data_init(mu,max_,min_)
    print(mu)
    # 采用随机初值
    # mu = np.random.rand(K, D)
    # print(mu)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha

def scale_data_init(Y,max_,min_):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y

def scale_data(Y):
    # 对每一维特征分别进行缩放
    max_ = []
    min_ = []
    for i in range(Y.shape[1]):
        max_.append(Y[:, i].max())
        min_.append(Y[:, i].min())
        temp_max = Y[:, i].max()
        temp_min = Y[:, i].min()
        Y[:, i] = (Y[:, i] - temp_min) / (temp_max - temp_min)
    return Y,temp_max,temp_min,max_,min_

# 给定迭代次数来优化
def GMM_EM(Y, K, times):
    Y,max_,min_,initmax,initmin = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K,max_,min_)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    return mu, cov, alpha,initmax,initmin

# 两次迭代间差值小于1e-8
def GMM_EM(Y, K):
    Y,max_,min_,initmax,initmin = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K,max_,min_)
    mu_pre = mu
    flag = False
    cnt = 0
    print(initmax,initmin)
    while(not(flag)):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
        temp = sum(sum(np.abs(mu-mu_pre)))
        if temp < 1e-8:
            flag = True
        mu_pre = mu
        cnt+=1
        print("当前迭代次数",cnt)
    return mu, cov, alpha,initmax,initmin

# 绘制最后的center，要从0-1之间恢复会数据集的状态
def recover(mu,initmax,initmin):
    mu_old = np.zeros(mu.shape)
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            mu_old[i][j] = mu[i][j]*(initmax[j]-initmin[j])+initmin[j]
    return mu_old


Y = data
matY = np.matrix(Y, copy=True)

# 计算参数
# mu, cov, alpha = GMM_EM(matY, K, 100)
mu, cov, alpha,initmax,initmin = GMM_EM(matY, K)

print('参数')
print("mu",mu)
print('------')
print("cov",cov)
print('---------')
print("alpha",alpha)
print('-------')

mu_old = recover(mu,initmax,initmin)
print(mu_old)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]
# 求当前模型参数下，各模型对样本的响应度矩阵
gamma = getExpectation(matY, mu, cov, alpha)
# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]
# 将每个样本放入对应类别的列表中
class1 = np.array([data[i] for i in range(N) if category[i] == 0])
class2 = np.array([data[i] for i in range(N) if category[i] == 1])
class3 = np.array([data[i] for i in range(N) if category[i] == 2])


# 绘制聚类结果
# done 把类的中心画出来（需要存极大极小）
plt.plot(class1[:, 2], class1[:, 3], 'co', label="class1")
plt.plot(class2[:, 2], class2[:, 3], 'yo', label="class2")
plt.plot(class3[:, 2], class3[:, 3], 'go', label="class3")
plt.plot(mu_old[:, 2], mu_old[:, 3], 'kp', label="center")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm Using K-means's result dim 2/3")
# plt.title("GMM Clustering By EM Algorithm dim 2/3")
plt.show()
plt.plot(class1[:, 0], class1[:, 1], 'co', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'yo', label="class2")
plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
plt.plot(mu_old[:, 0], mu_old[:, 1], 'kp', label="center")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm Using K-means's result dim 0/1")
# plt.title("GMM Clustering By EM Algorithm dim 0/1")
plt.show()

print("class1"," ",len(class1))
print("class2"," ",len(class2))
print("class3"," ",len(class3))

print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(data, category))
print("silhouette_scores", metrics.silhouette_score(data, category))