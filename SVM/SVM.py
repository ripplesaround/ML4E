# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/25 13:38
desc: 
'''
import sys
import copy
import data_process
import numpy as np
import matplotlib.pyplot as plt

class svm:
    def __init__(self,data,true_labels,K = 3):
        self.data = data
        self.true_labels = true_labels
        self.K = K  # 共有几类
        self.NUM = len(data)
        self.train_data = []
        self.test_data = []
        for i in range(self.K):
            temp = []
            temp1 = []
            temp_choice = np.random.choice(a=50,size = 40,replace = False)
            temp_choice += i * (self.NUM//self.K)
            for i in range(i*(self.NUM//self.K),(i+1)*self.NUM//self.K):
                if i in temp_choice:
                    temp.append(self.data[i])
                else:
                    temp1.append(self.data[i])
            print(len(temp),len(temp1))


