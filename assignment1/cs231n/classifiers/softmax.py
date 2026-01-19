from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax损失函数，朴素实现（带循环）。

    输入的维度为D，有C个类别，我们对N个样本的小批量进行操作。

    输入：
    - W: 一个形状为(D, C)的numpy数组，包含权重。
    - X: 一个形状为(N, D)的numpy数组，包含一个小批量的数据。
    - y: 一个形状为(N,)的numpy数组，包含训练标签；y[i] = c表示X[i]的标签为c，其中0 <= c < C。
    - reg: (float) 正则化强度

    返回一个元组：
    - 损失为一个浮点数
    - 相对于权重W的梯度；一个与W形状相同的数组
    """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    # 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # 以数值稳定的方式计算概率
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # 归一化
        logp = np.log(p)

        loss -= logp[y[i]]  # 负对数概率是损失

    # 归一化的hinge损失加上正则化
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # 计算损失函数的梯度并存储在dW中。                                           #
    # 与其先计算损失再计算导数，不如在计算损失的同时计算导数。                   #
    # 因此，您可能需要修改上面的某些代码以同时计算梯度。                         #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax损失函数，向量化版本。

    输入和输出与softmax_loss_naive相同。
    """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO:                                                                     #
    # 实现softmax损失的向量化版本，并将结果存储在loss中。                        #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # 实现softmax损失梯度的向量化版本，并将结果存储在dW中。                      #
    # 提示：与其从头计算梯度，不如重用您用于计算损失的一些中间值。                #
    #############################################################################

    return loss, dW
