# GNN综述

### 图神经网络 (GNN) 及认知推理

AI目前已经能够解决绝大多数的“感知（perceptron）”的工作

要逐步过渡到认知（cognition）的部分

<img src="GNN%E7%BB%BC%E8%BF%B0/image-20200413150758054.png" alt="image-20200413150758054" style="zoom:50%;" />

图神经网络：卷积神经网络+图模型

机器学习的算法在图上的延伸

- 如何在图上进行延伸

<img src="GNN%E7%BB%BC%E8%BF%B0/image-20200413154610582.png" alt="image-20200413154610582" style="zoom:50%;" />

- 点分类
- 连接生成
- 子图
- 网络相似度

**分布式表示**

#### 网络表示学习

- 网络的拓扑结构的复杂
  - 无上下、先后的概念
    - 对比图片、文本序列
  - 动态

##### deepwalk

word2vec

相同的单词意思（语境）比较相似，可以进行表示学习

在图上跑一个<u>random walk</u>，得到一个具有上下位序列的关系，然后再进行表示学习



表示学习究竟在做什么工作—> SVD分解 



#### GNN时代

太深层：过平滑 不鲁棒	Over-smoothing，non-robust

**邻居节点的表示放到图上**

GAT：影响力

卷积网络的本质又是什么呢？



如何把预训练与图挂钩？

<img src="GNN%E7%BB%BC%E8%BF%B0/image-20200413161911028.png" alt="image-20200413161911028" style="zoom:50%;" />



### 网络表示学习

传统的ml中数据点，点与点之间没有显式的联系（独立）

<img src="GNN%E7%BB%BC%E8%BF%B0/image-20200413232058798.png" alt="image-20200413232058798" style="zoom:50%;" />

##### Network Embedding

<img src="GNN%E7%BB%BC%E8%BF%B0/image-20200413233000849.png" alt="image-20200413233000849" style="zoom:50%;" />

核心是相似性的保持

如何定义网络间的相似性



Deepwalk

Node2Vec

LINE



网络： no 平移不变形

如何encode邻居节点

1. 谱方法

   网络映射到隐空间，类似于核方法？

   基于矩阵的分解，有好的理论基础

   矩阵太大不好分解，不能对应动态网络

2. 空间方法 spatial-based GCN

   可以搞动态网络



重构？重构误差？

用随机游走来重构这个图



**用图网络得到的结果对数据稀疏程度的依赖（影响）？**





## 参考文献

1. **唐杰** - 图神经网络 (GNN) 及认知推理 [链接](https://www.bilibili.com/video/BV1zp4y117bB/?spm_id_from=333.788.videocard.0)  [链接](https://event.baai.ac.cn/con/gnn-online-workshop-2020/ )

2. **宋国杰** - 网络表示学习 [链接](https://www.bilibili.com/video/BV1zp4y117bB/?spm_id_from=333.788.videocard.0)  [链接](https://event.baai.ac.cn/con/gnn-online-workshop-2020/ )

