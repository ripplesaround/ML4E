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

### 谱方法

#### 谱网络

谱是图的一种表示。

> The convolution operation is defined in the Fourier domain by computing the eigendecomposition of the graph Laplacian.

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417224437464.png" alt="image-20200417224437464" style="zoom:50%;" />

- 计算量大 intense computations

- 非空间的局部过滤器 non-spatially localized filters

  - 可以用平滑系数的方式将其过渡到局部空间化的

    > introducing a parameterization with smooth coefficients.

#### CHEBNET

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417225753480.png" alt="image-20200417225753480" style="zoom:50%;" />可以用Chebyshev polynomials进行截断展开（a truncated expansion）

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417225856663.png" alt="image-20200417225856663" style="zoom:50%;" />

#### GCN

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417232200148.png" alt="image-20200417232200148" style="zoom:50%;" />

图网络中的overfitting是如何表现的呢？

> overfitting on local neighborhood structures for graphs with very wide node degree distributions

可以被看作是谱方法，也可以被看作是空间方法

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417232510804.png" alt="image-20200417232510804" style="zoom:50%;" />

- 可能会有数值不稳定 / 梯度消失的特点
  - Renormalization trick
- 可以把x拓展，x是信号（singal），从一维拓展到多个维度，多个channel

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417232842455.png" alt="image-20200417232842455" style="zoom:50%;" />

#### AGCN - Adaptive Graph Convolution Network

原先的方法都只考虑了初始图的点之间的关系，自适应GCN是为了learn the underlying relations。

- AGCN学习了一个“剩余（residual）”图的拉普拉斯矩阵，并将其与原先的拉普拉斯矩阵结合。

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417233540566.png" alt="image-20200417233540566" style="zoom:50%;" />

  <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417233836053.png" alt="image-20200417233836053" style="zoom:50%;" />

  $\hat A$ 是一个学习到的邻接矩阵，is computed via a learned metric

  - 这个metric是提出的动机是欧式距离不能够衡量具有非欧性质的图网络

  - 所以采用了generalized Mahalanobis distance

    <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200417234244369.png" alt="image-20200417234244369" style="zoom:50%;" />

    （邻接矩阵可能存在太过于稀疏的问题？会有什么样的影响？）

  

### 空间方法 Spatital methods

谱方法的训练的过滤器基于拉普拉斯矩阵的特征分解，而这也取决于其中的图结构。也就说明谱方法无法直接拓展到不同结构的图上。

相对的，空间方法就把卷积操作直接定义到了图上

难点在于：

- 一个点的邻居的数量可能变化（differently sized neighborhoods）
- 如何保持CNN的局部不变性（local invariance）

#### NEURAL FPS

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418125522278.png" alt="image-20200418125522278" style="zoom:50%;" />

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418125600328.png" alt="image-20200418125600328" style="zoom:50%;" />

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418130542522.png" alt="image-20200418130542522" style="zoom:50%;" />

不同的权重矩阵对应不同度degree的点

- 缺点，对大图/有很多度的图不适用

#### PATCHY-SAN

选出K个邻居作为代表

- 选点序列 Node Sequence Selection

  这种方法不能处理整个图

  > It first uses a <u>graph labeling</u> procedure to get the order of the nodes and obtain the sequence of the nodes. Then the method uses a stride s to select nodes from the sequence until a number of w nodes are selected.

  随意游走产生序列？

  相当于把图（无序）转换成序列（有序）的结构，然后再选出K个点（。

- 聚集邻居点 Neighborhood Assembly

  构建 the receptive filed

  用bfs选出k个邻居

  > It first extracts the 1-hop neighbors of the node, then it considers high-order neighbors until the total number of k neighbors are extracted.

- **图正则化 Graph Normalization**

  给the receptive filed中的点一个排序

  从无序的图空间转换到向量空间（node2vec ？）

  >  assign nodes from two different graphs similar relative positions if they have similar structural roles.

- 卷积结构

  > The normalized neighborhoods serve as receptive fields and node and edge attributes are regarded as channels.

相当于是把非欧空间的问题转化到欧氏空间的问题

#### DCNN

diffusion-convolutional neural networks 扩散卷积

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418135932139.png" alt="image-20200418135932139" style="zoom:50%;" />

扩散是指考虑了K-hop？

#### DGCN

dual graph convolutional network 

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418145420948.png" alt="image-20200418145420948" style="zoom:50%;" />

一起考虑了局部一致性 local consistency和全局一致性 global consistency

用两个CNN分别捕捉局部/全局一致性并用无监督损失的方式将其组合在一起。

$Conv_A$ Local: <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418140757452.png" alt="image-20200418140757452" style="zoom:50%;" /> nearby nodes may have similar labels 

$Conv_P$ Global: <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418140826353.png" alt="image-20200418140826353" style="zoom:50%;" /> nodes with similar context may have similar labels

$X_p$  positive pointwise mutual information (PPMI) matrix

$D_p$ the diagonal degree matrix of $X_p$

the final loss function: <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418143604198.png" alt="image-20200418143604198" style="zoom:50%;" />

$L_0(Conv_A)$ 是关于给定node的label的监督的损失函数

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418145154493.png" alt="image-20200418145154493" style="zoom:50%;" />

Ground truth 真值



<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418145500245.png" alt="image-20200418145500245" style="zoom:50%;" />

是一种距离的衡量（无监督的/无label）

#### LGCN

Learnable graph convolutional networks

the sub-graph training strategy

采用max pooling来选出k个feature，然后用1-D CNN（？）来计算隐含的表示

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418152645688.png" alt="image-20200418152645688" style="zoom:50%;" />

$g(\cdot)$ the k-largest node selection operation 都是为了固定尺寸

$c(\cdot)$ the regular 1-D CNN

#### MONET

could generalize several previous techniques

compute <u>pseudo-coordinates</u> $\mathbf{u}(x,y)$ between the node and its neighbor and uses a weighting function among these coordinates

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418154916860.png" alt="image-20200418154916860" style="zoom:50%;" />

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418155254933.png" alt="image-20200418155254933" style="zoom:50%;" />

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418155144937.png" alt="image-20200418155144937" style="zoom:50%;" />

#### GRAPHSAGE

a general inductive framework

从点的局部邻居中取样并聚集特征来生成嵌入

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418160438392.png" alt="image-20200418160438392" style="zoom:50%;" />

aggregate 可以是

- mean
- lstm
- pooling

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418160725779.png" alt="image-20200418160725779" style="zoom:50%;" />

$P_n$ 负采样分布 negative sampling distribution？

$Q$ 负采样的数量



## CH06 GRN

用门技术（gate mechanism） 长期的信息传输

### GATED GRAPH NEURAL NETWORKS

#### GGNN

将GRU用在传递过程中

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418161955187.png" alt="image-20200418161955187" style="zoom:50%;" />

![image-20200418162918409](I2GNN%20%E7%AE%80%E6%98%8E/image-20200418162918409.png)

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418164140297.png" alt="image-20200418164140297" style="zoom:50%;" />

### Tree LSTM

- child-sum tree-LSTM

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418164217854.png" alt="image-20200418164217854" style="zoom:50%;" />

- N-ary Tree-LSTM

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418164707396.png" alt="image-20200418164707396" style="zoom:50%;" />

### GRAPH LSTM

graph-structured LSTM

### Sentence-LSTM

improving text encoding

regards each word as a node in the graph

Add a supernode

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418165451355.png" alt="image-20200418165451355" style="zoom:50%;" />

每个词节点从suprenode和邻居来聚集信息

每个supernode可以从supernode和所有word node来聚集信息

本质上还是global 和 local（context） 的结合



**还可以推广到ransfromer上**



## CH07 Graph Attention Networks

可以被看作是GCN一族中的一类

### GAT

graph attention network

在 propagation step 中考虑注意力机制

> It follows the self-attention strategy and the hidden state of each node is computed by attending over its neighbors.

任意点对的注意力系数：<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418211110671.png" alt="image-20200418211110671" style="zoom:50%;" />

每个点的最终的学到的特征： <img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418211510013.png" alt="image-20200418211510013" style="zoom:50%;" />

还可以搞 multi-head:

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200418212416612.png" alt="image-20200418212416612" style="zoom:50%;" />

特点：

- 关于点对的注意力参数的计算可以实现**并行**计算
- 可以处理不同度的点，并且可以给边赋上相关的权重
- 可以被应用到归纳学习（inductive learning problems）中



### GaAN

与GAN的区别：

> The difference between the attention aggregator in GaAN and the one in GAT is that GaAN uses the key-value attention mechanism and the dot product attention while GAT uses a fully connected layer to compute the attention coefficients.

在mutli-heads中，GaAN给不同的head不同的权重，用了一个额外的soft gate。即gated attention aggregator，用CNN和中心点+邻居生成gate value。



## CH08 Graph Residual Networks

unroll or stack layer 来实现k-hops的操作

实际操作中并不理想：可能会有很多噪声伴随着指数级增长的邻居节点

即使有了residual connections，多层GCN也不一定比2层的GCN高到哪里去

### HIGHWAY GCN

based on highway network

- layer-wise gates：能够让网络区分新 / 旧隐含状态

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419120312704.png" alt="image-20200419120312704" style="zoom:50%;" />

### JUMP KNOWLEDGE NETWORK

> different nodes in graphs may need different receptive fields for better representa- tion learning.

图的分布不均与，稀疏与dense

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419140550737.png" alt="image-20200419140550737" style="zoom:50%;" />

- learn <u>**adaptive**</u>, **<u>structure-aware</u>** representations

- 可以用来聚集信息的方法：
  - concatenation
  -  max-pooling
  - LSTM-attention

### DEEPGCNS

- vanishing gradient 
  - residual connections and dense connections from ResNet and DenseNet

- over smoothing
  - dilated convolutions

add skip connections

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419142000437.png" alt="image-20200419142000437" style="zoom:50%;" />

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419142058738.png" alt="image-20200419142058738" style="zoom:50%;" />

- *Dilated k-NN* method with dilation rate *d*

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419142416991.png" alt="image-20200419142416991" style="zoom:50%;" />

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419142427764.png" alt="image-20200419142427764" style="zoom:50%;" />

> The dilated convolution leverages information from different context and enlarges the receptive field of node v and is proven to be effective.

把不同距离的点聚集在一起相当于增加了到中心点的距离这一信息。



## CH09 Variants for Different Graph Types

### 有向图

DGP Dense Graph Propagation

### HETEROGENEOUS GRAPHS

异构图

### GRAPHS WITH EDGE INFORMATION



### DYNAMIC GRAPHS

<u>spatial</u> and <u>temporal</u> informatio

### MULTI-DIMENSIONAL GRAPHS



## CH10 Variants for Advanced Training Methods

### 采样

原有图上直接训练1.计算量大，2.缺乏the ability for inductive learning

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200419145046098.png" alt="image-20200419145046098" style="zoom:50%;" />

#### GraphSAGE

> GraphSAGE replaced full-graph Laplacian with learnable aggregation functions, which are key to perform message passing and generalize to <u>unseen nodes</u>.

random neighbor sampling 

> train the model via batches of nodes instead of the full-graph Laplacian

#### PinSage

是GraphSage在<u>图很大</u>的时候的拓展

importance-based sampling

random walk

> By simulating random walks starting from target nodes, this approach calculate the L1-normalized visit count of nodes visited by the random walk

### HIERARCHICAL POOLING

分级pooling



## CH11 General Frameworks

### MESSAGE PASSING NEURAL NETWORKS

用于监督学习

1. a message passing phase
2. a readout phase

#### The message passing phase （Propagation）

- a message function
- a vertex update function

<img src="I2GNN%20%E7%AE%80%E6%98%8E/image-20200420023619701.png" alt="image-20200420023619701" style="zoom:50%;" />

### NON-LOCAL NEURAL NETWORKS

capture long-range dependencies

CV

### GRAPH NETWORKS

还是直接看论文吧...