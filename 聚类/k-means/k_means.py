# 可以做一个关于数据集排序是否混乱的对比
import sys
import copy
import data_process
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
sys.path.append('../')

class kmeans:
    def __init__(self,path,K,orgin=None):
        if type(path)==str:
            self.path = path
            self.data, self.true_labels = data_process.read_file_list(self.path)
            self.orgin_data = self.data
        else:
            # print('hello\n\n')
            self.data = path
            self.orgin_data = orgin
        self.K = K
        # 数据集规模
        self.NUM = len(self.data)
        self.dim = (np.array(self.data)).shape[1]
        self.a = np.random.randint(0, self.NUM, self.K)
        print('读入数据完毕')
        self.label = []
        self.color = ['r', 'g', 'b']
        # 选定初始中心
        self.center = []
        self.dis = []
        self.iter_cnt = 1
        for i in range(self.K):
            self.label.append([])
        for i in self.a:
            self.center.append(self.data[i])
            self.dis.append(0)
        print("初始中心为", self.a)
        print(self.center)
        print("-----------------")

    def means(self,temp):
        ans = []
        dim = self.dim
        # print(dim)
        x1 = []
        for i in range(dim):
            x1.clear()
            for cnt in temp:
                x1.append(self.data[cnt][i])
            ans.append(np.mean(x1))
        return ans

    def classify(self,x):
        ans = 0
        for i in range(self.K):
            self.dis[i] = np.linalg.norm(x - self.center[i])
        min_index = self.dis.index(min(self.dis))
        # print(dis)
        return min_index

    def total_classify(self):
        fig = plt.figure()
        label_pre = copy.deepcopy(self.label)
        for i in range(self.K):
            self.label[i].clear()
        for temp in range(self.NUM):
            label_now = self.classify(self.data[temp])
            self.label[label_now].append(temp)
        if self.check(label_pre, self.label):
            return True
        return False

    def plot_label(self):
        for i in range(self.K):
            x = []
            y = []
            for cnt in self.label[i]:
                x.append(self.orgin_data[cnt][2])
                y.append(self.orgin_data[cnt][3])
            plt.scatter(x, y, c=self.color[i])
        plt.legend(['class1', 'class2', 'class3'], loc="best")
        # plt.title("K-means Algorithm dim 2/3")
        plt.title("dim 2/3")
        plt.show()
        for i in range(self.K):
            x = []
            y = []
            for cnt in self.label[i]:
                x.append(self.orgin_data[cnt][0])
                y.append(self.orgin_data[cnt][1])
            plt.scatter(x, y, c=self.color[i])
        plt.legend(['class1', 'class2', 'class3'], loc="best")
        # plt.title("K-means Algorithm dim 0/1")
        plt.title("dim 0/1")
        plt.show()
        y_pred = []
        for i in range(self.NUM):
            for j in range(self.K):
                if i in self.label[j]:
                    y_pred.append(j)
                    break
        # print(y_pred)
        print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(self.orgin_data, y_pred))

    def update_center(self):
        for i in range(self.K):
            self.center[i] = self.means(self.label[i])
        print("更新后的中心")
        print(self.center)

    def check(self,label_pre, label):
        for i in range(self.K):
            temp = (sorted(label[i]) == sorted(label_pre[i]))
            if temp == False:
                return False
        return True

    def cal(self):
        while (not self.total_classify()):
            print("第", self.iter_cnt, "循环")
            for i in range(self.K):
                print("第", i, "类中有", len(self.label[i]), "个数据")
            self.iter_cnt += 1
            self.update_center()
        self.plot_label()
    def plot_label_true(self):
        flag_pre = 0
        flag = 50
        for i in range(self.K):
            x = []
            y = []
            for cnt in range(flag_pre,flag):
                x.append(self.data[cnt][2])
                y.append(self.data[cnt][3])
            flag_pre = flag
            flag += 50
            plt.scatter(x, y,c=self.color[i])
        plt.legend(['class1', 'class2', 'class3'], loc="best")
        plt.title("dim 2/3")
        plt.show()
        flag_pre = 0
        flag = 50
        for i in range(self.K):
            x = []
            y = []
            for cnt in range(flag_pre,flag):
                x.append(self.data[cnt][0])
                y.append(self.data[cnt][1])
            plt.scatter(x, y,c=self.color[i])
            flag_pre = flag
            flag += 50
        plt.legend(['class1', 'class2', 'class3'], loc="best")
        plt.title("dim 0/1")
        plt.show()
