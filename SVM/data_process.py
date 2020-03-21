# coding=utf-8

'''
Author: ripples
Email: ripplesaround@sina.com

date: 2020/3/11 15:36
desc:
'''
import os
import numpy as np

def read_file_list(path, encoding='UTF-8'):
    results = []
    labels = []
    fin = open(path, 'r', encoding=encoding)
    for eachLiine in fin.readlines():
        line = eachLiine.strip().replace('\ufeff', '')
        cnt = []
        temp = line.split(',')
        for tem in temp[:-1]:
            cnt.append(float(tem))
        results.append(np.array(cnt))
        if temp[-1] == "Iris-setosa":
            labels.append(1)
        elif temp[-1] == "Iris-versicolor":
            labels.append(2)
        else:
            labels.append(3)
    fin.close()
    print('数据集中共有',len(results),'组数据')
    return results,labels

# 归一化
def scale(data):
    data = np.array(data)
    # row = len(data[0])
    # col = len(data)
    # results = []
    # for i in range(row):
    #     temp = []
    #     for j in range(col):
    #         temp.append(data[j][i])
    #     results.append(temp)
    # for i in range(len(results)):
    #     max_ = max(results[i])
    #     min_ = min(results[i])
    #     # results[i][] = (results[:, i] - min_) / (max_ - min_)
    for i in range(data.shape[1]):
        max_ = data[:, i].max()
        min_ = data[:, i].min()
        print(max_,min_)
        data[:, i] = (data[:, i] - min_) / (max_ - min_)
    return data

