import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def regionQuery(D, P, eps):
    """
    寻找在D中与点P距离在eps以内的邻近点。
    """
    neighbors = []
    for Pn in range(0, len(D)):
        # 如果距离在阈值以内，则将其添加到邻近点列表。
        if np.linalg.norm(D[P] - D[Pn]) <= eps:
            neighbors.append(Pn)
    return neighbors


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    参数:
      `D`      - 数据集（向量的列表）
      `labels` - 存储所有数据点的簇标签的列表
      `P`      - 新簇的种子点的索引
      `NeighborPts` - `P`的所有邻近点
      `C`      - 新簇的标签
      `eps`    - 阈值距离
      `MinPts` - 邻近点的最小要求数量
    """
    # 为种子点分配簇标签。
    labels[P] = C
    # NeighborPts将用作FIFO队列以搜索点
    # 在NeighborPts中，点由其在原始数据集中的索引表示。
    i = 0
    while i < len(NeighborPts):

        # 从队列中获取下一个点。
        Pn = NeighborPts[i]

        # 如果在种子搜索期间将Pn标记为NOISE，则知道它不是分支点（它没有足够的邻近点），所以
        # 将其作为簇C的叶子点并继续。
        if labels[Pn] == -1:
            labels[Pn] = C

        # 否则，如果Pn尚未被标记，请将其标记为C的一部分。
        elif labels[Pn] == 0:
            # 将Pn添加到簇C中（分配簇标签C）。
            labels[Pn] = C

            # 找到Pn的所有邻近点
            PnNeighborPts = regionQuery(D, Pn, eps)

            # 如果Pn至少有MinPts个邻近点，则它是一个分支点！
            # 将其所有邻近点添加到FIFO队列中以进行搜索。
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            # 如果Pn没有足够的邻近点，则它是一个叶子点。
            # 不要将其邻近点排队作为扩展点。
            # else:
            # 什么都不做
            # NeighborPts = NeighborPts

        # 在FIFO队列中前进到下一个点。
        i += 1
    return labels

    # 我们已完成簇C的扩展!


def MyDBSCAN(D, eps, MinPts):
    #    -1 - 表示噪声点
    #     0 - 表示点尚未被考虑。
    #     簇从1开始编号。
    # 最初所有标签都是0。
    labels = [0] * len(D)

    C = 0  # 记录簇的数量
    for P in range(0, len(D)):

        # 如果点的标签不是0，则继续到下一个点。
        if not (labels[P] == 0):
            continue

        # 找到P的所有邻近点。
        NeighborPts = regionQuery(D, P, eps)

        # 如果数量低于MinPts，假定该点是噪声。
        # NOISE点后来可能被某个簇捡起。
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # 否则，如果附近至少有MinPts个点，则使用此点作为新簇的种子。
        else:
            C += 1
            labels = growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
            # 所有数据已被聚类!
    return labels


if __name__ == "__main__":
    # 数据初始化
    m = 200
    data, y = make_moons(n_samples=m, random_state=123, noise=0.05)
    label = MyDBSCAN(data, 0.2, 5)
    label = np.array(label)
    plt.figure()
    for i in set(label):
        plt.scatter(data[label == i, 0], data[label == i, 1], marker='s')
    plt.show()
