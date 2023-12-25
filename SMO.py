import numpy as np

class SVM:
    def __init__(self, C, toler, max_iter):
        self.C = C  # 惩罚因子
        self.toler = toler  # 容错率
        self.max_iter = max_iter  # 最大迭代次数

    def fit(self, X, y):
        self.X = np.mat(X)
        self.y = np.mat(y).transpose()
        self.m, self.n = np.shape(X)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差缓存，第一列表示是否有效，第二列是误差值
        self.K = np.mat(np.zeros((self.m, self.m)))  # 存储核函数的计算结果
        for i in range(self.m):
            self.K[:, i] = self.kernelTrans(self.X, self.X[i, :])

        iter_num = 0
        entire_set = True
        alpha_pairs_changed = 0

        while (iter_num < self.max_iter) and ((alpha_pairs_changed > 0) or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(self.m):
                    alpha_pairs_changed += self.inner_loop(i)
                iter_num += 1
            else:
                non_bound_indices = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in non_bound_indices:
                    alpha_pairs_changed += self.inner_loop(i)
                iter_num += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True

        return self.b, self.alphas

    def inner_loop(self, i):
        Ei = self.calcEk(i)
        if ((self.y[i] * Ei < -self.toler) and (self.alphas[i] < self.C)) or \
           ((self.y[i] * Ei > self.toler) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaI_old = self.alphas[i].copy()
            alphaJ_old = self.alphas[j].copy()
            if self.y[i] != self.y[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                print("L == H")
                return 0

            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                print("eta >= 0")
                return 0

            self.alphas[j] -= self.y[j] * (Ei - Ej) / eta
            self.alphas[j] = self.clip_alpha(self.alphas[j], H, L)

            self.updateEk(j)

            if abs(self.alphas[j] - alphaJ_old) < 0.00001:
                print("j not moving enough")
                return 0

            self.alphas[i] += self.y[j] * self.y[i] * (alphaJ_old - self.alphas[j])
            self.updateEk(i)

            b1 = self.b - Ei - self.y[i] * (self.alphas[i] - alphaI_old) * self.K[i, i] - \
                 self.y[j] * (self.alphas[j] - alphaJ_old) * self.K[i, j]
            b2 = self.b - Ej - self.y[i] * (self.alphas[i] - alphaI_old) * self.K[i, j] - \
                 self.y[j] * (self.alphas[j] - alphaJ_old) * self.K[j, j]

            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            return 1
        else:
            return 0

    def kernelTrans(self, X, A, kTup=('lin', 0)):
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        if kTup[0] == 'lin':
            K = X * A.T
        elif kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * kTup[1] ** 2))
        else:
            raise NameError('The kernel is not recognized')
        return K

    def calcEk(self, k):
        fXk = float(np.multiply(self.alphas, self.y).T * self.K[:, k] + self.b)
        Ek = fXk - float(self.y[k])
        return Ek

    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(self.eCache[:, 0].A)[0]
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = self.selectJrand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    def selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj


if __name__ == "__main__":
    # 创建样本数据
    X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
    y = np.array([-1, -1, -1, 1, 1])

    # 创建SVM模型
    svm = SVM(C=1, toler=0.001, max_iter=500)

    # 训练模型
    b, alphas = svm.fit(X, y)

    print("b:", b)
    print("alphas:", alphas)
