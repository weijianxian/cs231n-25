from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    一个具有ReLU非线性和softmax损失的两层全连接神经网络，使用模块化层设计。我们假设输入维度为D，隐藏层维度为H，并对C类进行分类。

    架构应为 affine - relu - affine - softmax。

    注意，这个类不实现梯度下降；它将与一个单独的Solver对象交互，该对象负责运行优化。

    模型的可学习参数存储在字典self.params中，该字典将参数名称映射到numpy数组。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        初始化一个新的网络。

        输入：
        - input_dim: 一个整数，表示输入的大小
        - hidden_dim: 一个整数，表示隐藏层的大小
        - num_classes: 一个整数，表示分类的类别数
        - weight_scale: 标量，表示随机初始化权重的标准差
        - reg: 标量，表示L2正则化强度
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: 初始化两层网络的权重和偏置。权重应从以0.0为中心、标准差等于weight_scale的高斯分布中初始化，偏置应初始化为零。所有权重和偏置应存储在字典self.params中，第一层的权重和偏置使用键'W1'和'b1'，第二层的权重和偏置使用键'W2'和'b2'。
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        计算一小批数据的损失和梯度。

        输入：
        - X: 形状为(N, d_1, ..., d_k)的输入数据数组
        - y: 形状为(N,)的标签数组。y[i]给出X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播并返回：
        - scores: 形状为(N, C)的分类分数数组，其中scores[i, c]是X[i]和类别c的分类分数。

        如果y不为None，则运行训练时的前向和后向传播并返回一个元组：
        - loss: 标量值，表示损失
        - grads: 字典，与self.params具有相同的键，将参数名称映射到相对于这些参数的损失梯度。
        """
        scores = None
        ############################################################################
        # TODO: 实现两层网络的前向传播，计算X的分类分数并将其存储在scores变量中。
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 如果y为None，则我们处于测试模式，因此只返回scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: 实现两层网络的后向传播。将损失存储在loss变量中，将梯度存储在grads字典中。使用softmax计算数据损失，并确保grads[k]保存了self.params[k]的梯度。不要忘记添加L2正则化！
        # 注意：为了确保您的实现与我们的实现匹配并通过自动测试，请确保您的L2正则化包含一个0.5的因子以简化梯度表达式。
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """保存模型参数。"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """加载模型参数。"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


class FullyConnectedNet(object):
    """一个多层全连接神经网络的类。

    网络包含任意数量的隐藏层、ReLU非线性和一个softmax损失函数。这还将实现dropout和批量/层归一化作为选项。对于具有L层的网络，架构将是：

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中批量/层归一化和dropout是可选的，{...}块重复L - 1次。

    可学习参数存储在self.params字典中，并将使用Solver类学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """初始化一个新的FullyConnectedNet。

        输入：
        - hidden_dims: 一个整数列表，给出每个隐藏层的大小。
        - input_dim: 一个整数，表示输入的大小。
        - num_classes: 一个整数，表示分类的类别数。
        - dropout_keep_ratio: 介于0和1之间的标量，表示dropout强度。
            如果dropout_keep_ratio=1，则网络根本不应使用dropout。
        - normalization: 网络应使用什么类型的归一化。有效值为"batchnorm"，"layernorm"或None（默认值）。
        - reg: 标量，表示L2正则化强度。
        - weight_scale: 标量，表示随机初始化权重的标准差。
        - dtype: 一个numpy数据类型对象；所有计算都将使用此数据类型执行。float32速度更快但精度较低，因此您应使用float64进行数值梯度检查。
        - seed: 如果不为None，则将此随机种子传递给dropout层。
            这将使dropout层变得确定性，以便我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络的参数，将所有值存储在self.params字典中。将第一层的权重和偏置存储在W1和b1中；将第二层的权重和偏置存储在W2和b2中，依此类推。权重应从以0为中心、标准差等于weight_scale的正态分布中初始化。偏置应初始化为零。
        #                                                                          #
        # 当使用批量归一化时，第一层的缩放和偏移参数存储在gamma1和beta1中；第二层的缩放和偏移参数存储在gamma2和beta2中，依此类推。缩放参数应初始化为1，偏移参数应初始化为0。
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 当使用dropout时，我们需要向每个dropout层传递一个dropout_param字典，以便该层知道dropout概率和模式（训练/测试）。您可以将相同的dropout_param传递给每个dropout层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 使用批量归一化时，我们需要跟踪运行中的均值和方差，因此我们需要向每个批量归一化层传递一个特殊的bn_param对象。您应该将self.bn_params[0]传递给第一个批量归一化层的前向传递，将self.bn_params[1]传递给第二个批量归一化层的前向传递，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """计算全连接网络的损失和梯度。

        输入：
        - X: 形状为(N, d_1, ..., d_k)的输入数据数组
        - y: 形状为(N,)的标签数组。y[i]给出X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播并返回：
        - scores: 形状为(N, C)的分类分数数组，其中scores[i, c]是X[i]和类别c的分类分数。

        如果y不为None，则运行训练时的前向和后向传播并返回一个元组：
        - loss: 标量值，表示损失
        - grads: 字典，与self.params具有相同的键，将参数名称映射到相对于这些参数的损失梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 设置批量归一化参数和dropout参数的训练/测试模式，因为它们在训练和测试期间表现不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: 实现全连接网络的前向传播，计算X的分类分数并将其存储在scores变量中。
        #                                                                          #
        # 当使用dropout时，您需要将self.dropout_param传递给每个dropout的前向传播。
        #                                                                          #
        # 当使用批量归一化时，您需要将self.bn_params[0]传递给第一个批量归一化层的前向传播，
        # 将self.bn_params[1]传递给第二个批量归一化层的前向传播，依此类推。
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 如果是测试模式，则提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: 实现全连接网络的后向传播。将损失存储在loss变量中，将梯度存储在grads字典中。使用softmax计算数据损失，并确保grads[k]保存了self.params[k]的梯度。不要忘记添加L2正则化！
        #                                                                          #
        # 当使用批量/层归一化时，您不需要对缩放和偏移参数进行正则化。
        #                                                                          #
        # 注意：为了确保您的实现与我们的实现匹配并通过自动测试，请确保您的L2正则化包含一个0.5的因子以简化梯度表达式。
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
        """保存模型参数。"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """加载模型参数。"""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True
