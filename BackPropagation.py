import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return x * (1 - x)


# 定义对应的tanh(x)函数及在BP算法中对应的导数
def tanh(x):
    return (-np.exp(-x) + np.exp(x)) / (np.exp(x) + np.exp(-x))


def tanh_deriv(x):
    return 1 - x ** 2


class NeuralNetwork:
    def __init__(self, layers, activation='logistic'):
        """
        layers: 包含每个层中单元数量的列表，至少应包含两个值。
        activation: 要使用的激活函数。可以是 "logistic" 或 "tanh"。
        """

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):  # 初始化层之间的权重
            # gaussian分布生成随机矩阵
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))
        # 接上最后一层，输出无截距添加
        self.weights.append(np.random.randn(layers[i], layers[i + 1]))

    def fit(self, X, y, learning_rate=0.1, epochs=10000):
        X = np.atleast_2d(X)
        y = np.array(y)

        for k in range(epochs):  # epchos代表迭代次数
            i = np.random.randint(X.shape[0])  # 随机选取样本中的任意一个样本点
            a = [X[i]]  # 1*特征维数的向量

            for l in range(len(self.weights)):  # 前馈神经网络，计算每个神经元的数值
                # 从l层计算l+1层神经元参数，a(l+1)=f(x(l)W(l)),其中f为激活函数
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]  # 计算最后一层的误差，即e(最后一层)
            # 计算delta(最后一层)=e(最后一层)*h(a(最后一层)) h为对应激活函数的导数
            deltas = [error * self.activation_deriv(a[-1])]

            # 开始BP算法
            for l in range(len(a) - 2, 0, -1):  # 从倒数第二层开始计算delta
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(
                    a[l]))  # delta(l)=(delta(l+1)*W(l)的转置).*h(a(l))
            deltas.reverse()  # 将delta按照前馈神经网络顺序排列
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                # 每层更新的偏导数为a[i]的转秩*delta[i]
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):  # 针对进来的样本数据通过训练好的网络进行预测
        x = np.array(x)
        a = x
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
