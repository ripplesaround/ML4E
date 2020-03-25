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

class svm:
    def __init__(self,data,true_labels,K = 3,train_size=40):
        self.data = data
        self.true_labels = true_labels
        self.K = K  # 共有几类
        self.NUM = len(data)
        self.train_data = []
        self.test_data = []
        for i in range(self.K):
            temp = []
            temp1 = []
            temp_choice = np.random.choice(a=(self.NUM//self.K),size = train_size,replace = False)
            temp_choice += i * (self.NUM//self.K)
            for i in range(i*(self.NUM//self.K),(i+1)*self.NUM//self.K):
                if i in temp_choice:
                    temp.append(self.data[i])
                else:
                    temp1.append(self.data[i])
            self.train_data.append(np.array(temp))
            self.test_data.append(np.array(temp1))
        self.train_size = train_size
