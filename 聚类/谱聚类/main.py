# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/17 22:19
desc:
'''
import sys
import copy
sys.path.append("../k-means")
sys.path.append("../")
import data_process
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from Spectral_Clustering import *

K = 3
# 数据集规模
N = 150
path = '../iris/iris.data'
data, true_labels= data_process.read_file_list(path)
data = np.array(data)
print('读入数据完毕')

test = mysc(3,data)
test.cluster_plot()

# test =sk_sc(3,data)
# test.cal_plot()



# todo 探究gamma的值对结果的影响
# 0.01 0.1 1 10
# todo 探究取前k大还是前k小

# todo 可以加入衡量指标！加上更好，不加也可以
# https://www.cnblogs.com/pinard/p/6235920.html
# https://blog.csdn.net/sinat_26917383/article/details/70577710
# 数据挖掘导论（完整版）里面也有