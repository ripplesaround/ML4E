# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/11 15:34
desc:
'''
# 可以做一个关于数据集是否混乱的对比

import sys
import copy
import data_process
import numpy as np
import matplotlib.pyplot as plt
from k_means import kmeans
sys.path.append('../')
path = '../iris/iris.data'
x = kmeans(path,3)
x.cal()
# x.plot_label_true()
