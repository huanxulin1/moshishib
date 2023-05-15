import numpy as np
from log import log_creater

def classify_KNN(X_train, Y_train, X_test, Y_test, k):
    # 设定一个三维数组a[i][j][k]，i代表测试样本个数，j代表训练样本个数，k为2
    # a[i][j][0]表示第i个测试样本点到第j个训练样本点的距离，a[i][j][1]表示第j个训练样本的标签
    distance = np.zeros((len(X_test), len(X_train), 2))
    # 设定y_pred为测试样本的标签数组
    y_pred = np.zeros(len(X_test))
    # 计算每个测试样本到每个训练样本的欧式距离和对应训练样本标签
    for i in range(len(X_test)):
        # 计算欧氏距离
        for j in range(len(X_train)):
            dist = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))
            distance[i][j][0] = dist
            distance[i][j][1] = Y_train[j]
            # print(distance[i][j])
        # print(distance[i].shape)
        # 使用冒泡排序使得距离从小到大
        for m in range(len(X_train)):
            for n in range(0, len(X_train) - i - 1):
                if distance[i][n][0] > distance[i][n + 1][0]:
                    distance[i][n], distance[i][n + 1] = distance[i][n + 1], distance[i][n]
        # 取出距离最小的k个点
        k_dis = distance[i][:k]
        # print(k_dis)
        count = 0
        # 计算测试样本点距离标签为1的训练样本点最小的个数
        for l in range(k):
            if k_dis[l][1] == 1:
                count += 1
        # 分类，概率最高的标签为测试样本点的标签
        if (count / k) > ((k - count) / k):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    # print(y_pred)
    # 计算预测正确的样本数
    count = 0
    for i in range(len(X_test)):
        if y_pred[i] == Y_test[i]:
            count += 1
    # 计算准确率
    acc = count / len(X_test)
    return y_pred, acc

def KNN_train(X_train, Y_train, X_test, Y_test, K):
    acc_list = []
    # 创建日志文件
    logger = log_creater('work_dir/KNN')
    for k in K:
        #不同k值进行训练
        logger.info("k = %d" % k)
        logger.info("start KNN")
        _, acc = classify_KNN(X_train, Y_train, X_test, Y_test, k)
        logger.info("acc = %.3f" % acc)
        acc_list.append(acc)
    return

class SOM(object):

    _train = False

    def __init__(self, m, n, dim, iter = 100, alpha = None, sigma = None):
        self.m = m
        self.n = n
        self.dim = dim

        if alpha == None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma == None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)

        self.iter = abs(int(iter))
