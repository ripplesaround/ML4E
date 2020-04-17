# Introduction to Graph Neural Networks

《Introduction to Graph Neural Networks》 Zhiyuan Liu and Jie Zhou

此书的简明的“翻译”，旨在速读后掌握GNN的相关知识。

---



## CH01 Introduction

图：object 点，relationship 边

一种非欧氏的数据

- 点分类
- 预测
- 聚类

### 引入图网络的动机

1. 卷积神经网络
   1. 局部链接

   2. 共享权重

      可以减少传统基于谱的图理论的计算量

   3. 多层网络

2. 网络的嵌入 / 表示 graph embedding

   1. 没有参数共享
   2. 直接嵌入缺乏泛化的能力，不能对付动态的网络结构



## CH02 数学基础

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200415011906579.png" alt="image-20200415011906579" style="zoom:50%;" />

### 图的数学理论

adjacent 毗邻

- 邻接矩阵

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180113300.png" alt="image-20200417180113300" style="zoom:50%;" />

- 度矩阵

  - degree： the number of edges connected with v

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180122314.png" alt="image-20200417180122314" style="zoom:50%;" />

- Laplcaian Matrix 拉普拉斯矩阵

  定义在无向图中

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180137187.png" alt="image-20200417180137187" style="zoom: 45%;" />

- Symmetric normalized Laplacian

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180424733.png" alt="image-20200417180424733" style="zoom:50%;" />

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180436741.png" alt="image-20200417180436741" style="zoom:50%;" />

- Random walk normalized Laplacian

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180505429.png" alt="image-20200417180505429" style="zoom:45%;" />

- Incidence matrix 关联矩阵

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417180744193.png" alt="image-20200417180744193" style="zoom:50%;" />



## CH03 神经网络基础

激活函数

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417181018517.png" alt="image-20200417181018517" style="zoom:50%;" />

反向传播



## CH04 普通/传统（Vanilla）的GNN

> A node is naturally defined by its features and related nodes in the graph.

> The target of GNN is to learn a state embedding $h_v \in R^s$, which encodes the information of the neighbor- hood, for each node.

目的是得到状态的嵌入，然后用其来处理下游任务

#### local transition fuction / local output function

f / g

所有点共享的函数（类似于CNN的窗口？）

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417191906740.png" alt="image-20200417191906740" style="zoom:50%;" />

可以写成矩阵形式（把所有的状态都压在一起）

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417195556003.png" alt="image-20200417195556003" style="zoom:50%;" />

contraction map 压缩映射

不动点理论（Banach’s fixed point theorem）

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417200149462.png" alt="image-20200417200149462" style="zoom:50%;" />



## CH05 GCN

GCN主要进展两大方向

- 谱方法 Spectral approach
- 空间方法 spatial approach

