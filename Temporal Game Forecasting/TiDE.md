# Long-term Forecasting with TiDE: Time-series Dense Encoder

This document is a summary of the paper [Long-term Forecasting with TiDE: Time-series Dense Encoder] by GCYYfun.

## Introduction

### 原文

Long-term forecasting, which is to predict several steps into the future given a long context or look-back, is one of the most fundamental problems in time series analysis, with broad applications in energy, finance, and transportation. Deep learning models [WXWL21, NNSK22] have emerged as the preferred approach for forecasting rich, multivariate, time series data, outperforming classical statistical approaches such as ARIMA or GARCH [BJRL15]. In several forecasting competitions such as the M5 competition [MSA20] and IARAI Traffic4cast contest [KKJ+20], almost all the winning solutions are based on deep neural networks.

长期预测,也就是给定长时间的上下文或回顾后预测未来若干步骤,是时间序列分析中最基本的问题之一,在能源、金融和交通等领域有广泛应用。深度学习模型[WXWL21, NNSK22]已经成为预测复杂的、多变量的时间序列数据的首选方法,优于经典的统计方法如ARIMA或GARCH [BJRL15]。在M5预测竞赛[MSA20]和IARAI Traffic4cast竞赛[KKJ+20]等几场预测竞赛中,几乎所有获胜的解决方案都是基于深度神经网络的。

----
Note：  
提到的首选方法？mark一下之后了解。

[WXWL21] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419–22430, 2021.

[NNSK22] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. International conference on learning representations, 2022.

---

Various neural network architectures have been explored for forecasting, ranging from recurrent neural networks to convolutional networks to graph-neural-networks. For sequence modeling tasks in domains such as language, speech and vision, Transformers [VSP+17] have emerged as the most successful deep learning architecture, even outperforming recurrent neural networks (LSTMs)[HS97]. Subsequently, there has been a surge of Transformer-based forecasting papers [WXWL21, ZZP+21, ZMW+22] in the time-series community that have claimed state-of-the-art (SoTA) forecasting performance for long-horizon tasks. However recent work [ZCZX23] has shown that these Tranformers-based architectures may not be as powerful as one might expect for time series forecasting, and can be easily outperformed by a simple linear model on forecasting benchmarks.
Such a linear model however has de ciencies since it is ill-suited for modeling non-linear dependencies among the time-series sequence and the time-independent covariates. Indeed, a very recent paper [NNSK22] proposed a new Transformer-based architecture that obtains SoTA performance for deep neural networks on the standard multivariate forecasting benchmark.


各种神经网络架构已经被探索用于预测,包括递归神经网络、卷积网络和图神经网络。在语言、语音和视觉等序列建模任务中,Transformer [VSP+17]已经成为最成功的深度学习架构,甚至优于递归神经网络(LSTM)[HS97]。随后,时间序列社区出现了大量基于Transformer的预测论文[WXWL21, ZZP+21, ZMW+22],声称在长期预测任务上取得了最先进(SoTA)的预测性能。然而,最新的工作[ZCZX23]表明,这些基于Tranformer的架构在时间序列预测中可能没有人们期待的那么强大,可以被一个简单的线性模型轻松打败。
但是,这样一个线性模型也有缺陷,因为它不适合建模时间序列和时间无关的协变量之间的非线性依赖关系。实际上,一篇最新的论文[NNSK22]提出了一个新的基于Transformer的架构,在标准的多变量预测基准测试中,取得了深度神经网络最先进的性能。

In this paper, we present a simple and effective deep learning architecture for forecasting that obtains superior performance when compared to existing SoTA neural network based models on the long-term time series forecasting benchmarks. Our Multi-Layer Perceptron (MLP)-based model is embarrassingly simple without any self-attention, recurrent or convolutional mechanism. Therefore, it enjoys a linear computational scaling in terms of the context and horizon lengths unlike many Transformer based solutions.

在这篇论文中,我们提出了一个简单有效的深度学习架构用于预测,相比现有的基于神经网络的最先进模型,在长期时间序列预测基准测试中获得了优越的性能。我们的多层感知机(MLP)基模型非常简单,没有任何自注意力、递归或卷积机制。因此,与许多基于Transformer的解决方案不同,它在计算复杂度上与上下文和预测horizon长度呈线性关系。

----
Note：  

*这样一个线性模型也有缺陷,因为它不适合建模时间序列和时间无关的协变量之间的非线性依赖关系*  

那这种结构不适合博弈吧...

---

The main contributions of this work are as follows:

• We propose the Time-series Dense Encoder (TiDE) model architecture for long-term time
series forecasting. TiDE encodes the past of a time-series along with covariates using dense MLPs and then decodes time-series along with future covariates, again using dense MLPs.

• We analyze the simplest linear analogue of our model and prove that this linear model can achieve near optimal error rate in linear dynamical systems (LDS) [Kal63] when the design matrix of the LDS has maximum singular value bounded away from 1. We empirically verify this on a simulated dataset where the linear model outperforms LSTMs and Transformers.

• On popular real-world long-term forecasting benchmarks, our model achieves better or similar performance compared to prior neural network based baselines (>10% lower Mean Squared Error on the largest dataset). At the same time, TiDE is 5x faster in terms of inference andmore than 10x faster in training when compared to the best Transformer based model.

本文的主要贡献如下:

- 我们提出了Time-series Dense Encoder (TiDE)模型架构用于长期时间序列预测。TiDE使用密集的MLP对时间序列的过去以及协变量进行编码,然后使用密集的MLP对时间序列未来以及未来的协变量进行解码。

- 我们分析了我们模型的最简单的线性模拟,并证明当线性动力系统(LDS)[Kal63]的设计矩阵最大奇异值远离1时,这个线性模型可以在LDS中达到近似最优误差率。我们在一个模拟数据集上经验证明了这一点,线性模型超过了LSTM和Transformer。

- 在流行的真实世界长期预测基准测试中,我们的模型与之前基于神经网络的基准相比获得了更好或相似的性能(在最大数据集上Mean Squared Error降低了10%以上)。与此同时,TiDE在推理速度上比最佳的基于Transformer的模型快5倍,在训练速度上快10倍以上。



## 2 背景和相关工作
长期预测的模型可以大致分为多变量模型和单变量模型。

多变量模型接受所有 相关时间序列变量的过去,并作为这些过去的联合函数预测所有时间序列的未来。

这包括经典的VAR模型[ZW06]。我们将主要关注基于神经网络的长期预测的先前工作。  
LongTrans [LJX+19]使用LogSparse设计的注意力层来捕获局部信息,其空间和计算复杂度接近线性。  
Informer [ZZP+21]使用ProbSparse自注意力机制来实现对上下文长度的次二次方依赖性。  
Autoformer [WXWL21]使用趋势和季节性分解以及次二次方自注意力机制。  
FEDFormer [ZMW+22]使用频率增强结构,而  
Pyraformer [LYL+21]使用线性复杂度的金字塔自注意力,可以关注不同粒度。

另一方面,单变量模型将时间序列变量的未来预测为仅该时间序列的过去和协变量特征的函数,即其他时间序列的过去不是推理时的输入的一部分。  
单变量模型有两种,局部的和全局的。局部单变量模型通常每个时间序列变量训练一次,推理也是每个时间序列进行一次。经典模型如AR、ARIMA、指数平滑模型[McK84]和Box-Jenkins方法论[BJ68]都属于这一类别。我们建议读者阅读[BJRL15],深入了解这些方法。

全局单变量模型由一个共享模型组成,它接受一个时间序列的过去(以及协变量)来预测其未来。但是,模型的权重在整个数据集上得到联合训练。这一类别主要包括基于深度学习的架构,如[SFGJ20]。

在长期预测的背景下,最近观察到一个简单的线性全局单变量模型可以在长期预测中打败基于transformer的多变量方法[ZCZX23]。Dlinear [ZCZX23]学习从上下文到预测horizon的线性映射,指出自注意力机制的次二次方逼近的缺陷。事实上,一个最新的模型PatchTST [NNSK22]已经证明,将时间序列的连续patch作为token馈送到 vanilla自注意力机制中,可以在长期预测基准中打败DLinear的性能。

注意,如果在同一任务的同一测试集上进行评估,则所有类别的模型都可以在多变量长期预测任务上进行公平比较,这也是我们在第6节中遵循的协议。

## 3 问题设置

在描述问题设置之前,我们需要建立一些通用符号。

###  3.1 符号表示

我们用粗体大写字母表示矩阵,如X ∈ R^{N×T}。  
切片表示法 i:j 表示集合 {i, i+1, ..., j},[n] := {1, 2, ..., n}。  
除非另有说明,否则各行和列始终被视为列向量。  
我们也可以使用集合来选择子矩阵,即 X[I; J] 表示选择行在I内、列在J内的子矩阵。  
X[:, j]表示选择第j列,X[i, :]表示第i行。  
表示法[v; u]将表示两个列向量的连接,相同的表示法也可用于沿维度的矩阵连接。  

### 3.2 多变量预测

在本节中,我们首先抽象出长期多变量预测的核心问题。

数据集中有N个时间序列。第i个时间序列的look-back将表示为y(i)_1:L,而预测horizon表示为y(i)_L+1:L+H。预测者的任务是在给定look-back的情况下预测horizon时间点。

在许多预测场景中,可能会提前知道动态和静态协变量。稍微滥用符号表示法,我们将使用x(i)_t ∈ R^r来表示时间序列i在时间t的r维动态协变量。例如,它们可以是全局协变量(对所有时间序列共同),如星期、假日等,或者对于某个时间序列特定的,例如在需求预测用例中某天某产品的折扣。我们也可以有时间序列的静态属性a(i),如零售需求预测中不随时间变化的产品特征。在许多应用中,这些协变量对于准确预测至关重要,一个好的模型架构应该能处理它们。

预测者可以看作是一个函数 $f$,它将以下内容映射到未来的准确预测:

每个时间序列$i$的历史$y^{(i)}_{1:L}$
每个时间序列$i$的动态协变量$x^{(i)}_{1:L+H}$
每个时间序列$i$的静态属性$a^{(i)}$

$$f: ({\{y^{(i)}_{1:L}\ }\}_{i=1}^N , {\{ x^{(i)}_{1:L+H}\ }\}_{i=1}^N , {\{a^{(i)}}\}_{i=1}^N) \rightarrow {\{\hat{y}^{(i)}_{L+1:L+H}}\}_{i=1}^N$$

预测的准确性将通过一个量化预测值和实际值接近程度的指标来测量。例如,如果指标是均方误差(MSE),那么拟合好坏的测量公式为:

