# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/25 13:39
desc: 
'''
import sys
import copy
import data_process
import numpy as np
import matplotlib.pyplot as plt
from SVM import *

K = 3
# 数据集规模
N = 150
path = 'iris/iris.data'
data, true_labels= data_process.read_file_list(path)
data = np.array(data)
print('读入数据完毕')

test = svm_oago(data, true_labels)
test.train_oago()

# test = svm_oagm(data, true_labels)
# test.train_oagm()