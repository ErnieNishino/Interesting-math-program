import numpy as np


def onehot(targets, num):
    """
    将目标标签列表转换为一热编码格式。

    参数：
    - targets: 目标标签列表
    - num: 样本数量（targets列表的长度）

    返回：
    - result: 一热编码矩阵，其中每行对应一个目标标签
    """

    # 创建一个维度为(num, 10)的零矩阵
    result = np.zeros((num, 10))

    # 遍历目标标签，将每行中对应的元素设置为1
    for i in range(num):
        result[i][targets[i]] = 1

    return result


def img2col(x, ksize, stride):
    """
    将图像矩阵信息转换为列格式，以便与卷积核进行矩阵乘法。

    参数：
    - x: 输入图像矩阵
    - ksize: 卷积核的大小
    - stride: 卷积操作的步长

    返回：
    - image_col: 列格式的矩阵，适用于卷积操作
    """

    # 获取输入图像的维度
    wx, hx, cx = x.shape

    # 计算卷积后特征图的宽度
    feature_w = (wx - ksize) // stride + 1

    # 创建一个维度为(feature_w*feature_w, ksize*ksize*cx)的零矩阵
    image_col = np.zeros((feature_w * feature_w, ksize * ksize * cx))

    # 初始化一个变量，用于跟踪image_col矩阵中的行索引
    num = 0

    # 遍历特征图
    for i in range(feature_w):
        for j in range(feature_w):
            # 从输入图像中提取一个区域，并将其重塑为列
            image_col[num] = x[i * stride:i * stride + ksize, j * stride:j * stride + ksize, :].reshape(-1)
            num += 1

    # 返回列格式的矩阵，适用于卷积操作
    return image_col


# nn全连接神经网络部分
class Linear(object):
    def __init__(self, inChannel, outChannel):
        """
        初始化具有随机权重和偏置的线性层。

        参数：
        - inChannel：输入神经元数量。
        - outChannel：输出神经元数量。
        """
        scale = np.sqrt(inChannel / 2)

        # 使用标准正态分布和缩放因子随机初始化权重
        self.W = np.random.standard_normal((inChannel, outChannel)) / scale

        # 使用标准正态分布和缩放因子随机初始化偏置
        self.b = np.random.standard_normal(outChannel) / scale

        # 初始化权重和偏置的梯度矩阵
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)

    def forward(self, x):
        """
        通过线性层进行前向传播。

        参数：
        - x：输入数据。

        返回：
        - x_forward：线性层的输出。
        """
        self.x = x

        # 计算线性变换（点积）并加上偏置
        x_forward = np.dot(self.x, self.W) + self.b

        return x_forward

    def backward(self, delta, learning_rate):
        """
        通过线性层进行反向传播。

        参数：
        - delta：来自下一层的梯度。
        - learning_rate：用于更新权重和偏置的学习率。

        返回：
        - delta_backward：传递到上一层的梯度。
        """
        # 使用批量梯度下降计算梯度
        batch_size = self.x.shape[0]
        self.W_gradient = np.dot(self.x.T, delta) / batch_size
        self.b_gradient = np.sum(delta, axis=0) / batch_size

        # 计算传递到上一层的梯度
        delta_backward = np.dot(delta, self.W.T)

        # 使用梯度下降更新权重和偏置
        self.W -= self.W_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


class Conv(object):
    def __init__(self, kernel_shape, stride=1, pad=0):
        """
        初始化卷积层。

        参数：
        - kernel_shape: 卷积核的形状 (width, height, in_channel, out_channel)
        - stride: 卷积的步长，默认为1
        - pad: 卷积的填充，默认为0
        """
        width, height, in_channel, out_channel = kernel_shape
        self.stride = stride
        self.pad = pad
        scale = np.sqrt(3 * in_channel * width * height / out_channel)  # 标准化
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(out_channel) / scale
        self.k_gradient = np.zeros(kernel_shape)
        self.b_gradient = np.zeros(out_channel)

    def forward(self, x):
        """
        卷积层的前向传播。

        参数：
        - x: 输入数据。

        返回：
        - feature: 卷积层的输出。
        """
        self.x = x

        # 如果有填充，则对输入数据进行填充
        if self.pad != 0:
            self.x = np.pad(self.x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant')

        bx, wx, hx, cx = self.x.shape
        wk, hk, ck, nk = self.k.shape
        feature_w = (wx - wk) // self.stride + 1
        feature = np.zeros((bx, feature_w, feature_w, nk))

        self.image_col = []
        kernel = self.k.reshape(-1, nk)

        # 对每个样本进行卷积操作
        for i in range(bx):
            image_col = img2col(self.x[i], wk, self.stride)
            feature[i] = (np.dot(image_col, kernel) + self.b).reshape(feature_w, feature_w, nk)
            self.image_col.append(image_col)

        return feature

    def backward(self, delta, learning_rate):
        """
        卷积层的反向传播。

        参数：
        - delta: 从后一层传递过来的梯度。
        - learning_rate: 学习率，用于更新权重和偏置。

        返回：
        - delta_backward: 传递到前一层的梯度。
        """
        bx, wx, hx, cx = self.x.shape
        wk, hk, ck, nk = self.k.shape
        bd, wd, hd, cd = delta.shape

        # 计算权重和偏置的梯度
        delta_col = delta.reshape(bd, -1, cd)
        for i in range(bx):
            self.k_gradient += np.dot(self.image_col[i].T, delta_col[i]).reshape(self.k.shape)
        self.k_gradient /= bx
        self.b_gradient += np.sum(delta_col, axis=(0, 1))
        self.b_gradient /= bx

        # 计算传递到前一层的梯度
        delta_backward = np.zeros(self.x.shape)
        k_180 = np.rot90(self.k, 2, (0, 1))
        k_180 = k_180.swapaxes(2, 3)
        k_180_col = k_180.reshape(-1, ck)

        # 如果输入数据和梯度的尺寸不一致，通过填充补全delta
        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

        for i in range(bx):
            pad_delta_col = img2col(pad_delta[i], wk, self.stride)
            delta_backward[i] = np.dot(pad_delta_col, k_180_col).reshape(wx, hx, ck)

        # 反向传播，更新权重和偏置
        self.k -= self.k_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


class Pool(object):
    def forward(self, x):
        """
        池化层的前向传播。

        参数：
        - x: 输入数据。

        返回：
        - feature: 池化层的输出。
        """
        b, w, h, c = x.shape
        feature_w = w // 2
        feature = np.zeros((b, feature_w, feature_w, c))
        self.feature_mask = np.zeros((b, w, h, c))

        for bi in range(b):
            for ci in range(c):
                for i in range(feature_w):
                    for j in range(feature_w):
                        feature[bi, i, j, ci] = np.max(x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        index = np.argmax(x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        self.feature_mask[bi, i * 2 + index // 2, j * 2 + index % 2, ci] = 1
        return feature

    def backward(self, delta):
        """
        池化层的反向传播。

        参数：
        - delta: 从后一层传递过来的梯度。

        返回：
        - delta_backward: 传递到前一层的梯度。
        """
        return np.repeat(np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask


# Relu 激活函数
class Relu(object):
    def forward(self, x):
        """
        Relu激活函数的前向传播。

        参数：
        - x: 输入数据。

        返回：
        - output: 激活后的输出。
        """
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        """
        Relu激活函数的反向传播。

        参数：
        - delta: 从后一层传递过来的梯度。

        返回：
        - delta_backward: 传递到前一层的梯度。
        """
        delta[self.x < 0] = 0
        return delta


# Softmax 数据量大时可以设置batchsize减少梯度计算量
class Softmax(object):
    def cal_loss(self, predict, label):
        """
        计算Softmax损失函数。

        参数：
        - predict: 预测概率。
        - label: 真实标签。

        返回：
        - loss: 损失值。
        - delta: 损失对预测概率的梯度。
        """
        batchsize, classes = predict.shape
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)

        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])

        loss /= batchsize
        return loss, delta

    def predict(self, predict):
        """
        Softmax激活函数的前向传播。

        参数：
        - predict: 预测概率。

        返回：
        - softmax: 归一化后的概率。
        """
        batchsize, classes = predict.shape
        self.softmax = np.zeros(predict.shape)

        for i in range(batchsize):
            predict_tmp = predict[i]
            predict_tmp = np.exp(predict_tmp)
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)

        return self.softmax
