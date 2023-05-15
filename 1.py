import numpy as np
import pandas as pd

#--------------------------EM算法------------------------------#
#求高斯概率密度函数值
def gaussian(X, mean, cov):
    #由于数据是一维的，那么不需要考虑多个特征
    t = pow((X- mean), 2)
    cope = 1.0 / np.sqrt(2 * np.pi * cov)
    cove = cope * np.exp(-0.5 / cov * t)
    return cove

def EM(X, K, max_iter):
    m = len(X)
    X_mean = np.sum(X) / m
    X_cov = 0
    for i in range(m):
        X_cov += pow((X[i] - X_mean), 2)
    X_cov /= m
    means = np.zeros(K)
    means[0] = min(X)
    means[1] = max(X)
    covs = np.zeros(K)
    covs[0] = np.identity(1)
    covs[1] = np.identity(1)
    pirors = np.zeros(K)
    pirors[0] = pirors[1] = 0.5

    #迭代max_iter次
    for iter in range(max_iter):
        #E-step
        gamma = np.zeros((m, K))
        for i in range(m):
            for j in range(K):
                gamma[i, j] = pirors[j] * gaussian(X[i], means[j], covs[j])
            t = np.sum(gamma[i])
            #print(t)
            gamma[i] /= t

        #M-step 更新参数
        for j in range(K):
            N = np.sum(gamma[:, j])
            #means[j] = np.sum(X * gamma[:, j].reshape(m, 1)) / N
            means[j] = 0
            for i in range(m):
                means[j] += (X[i] * gamma[i, j])
            means[j] /= N
            covs[j] = 0
            for i in range(m):
                covs[j] += gamma[i, j] * ((X[i] - means[j]) ** 2)
            covs[j] /= N
            pirors[j] = N / m
    return means, covs, pirors

#--------------------朴素贝叶斯算法-------------------------#

def handle_data(data):
    # 初始化类别数据字典
    cate_dict = {}
    # 数据集表头列表（各个条件及分类结果）
    header_list = data.columns.tolist()
    # 条件列表
    factor_list = header_list[:-1]
    # 分类结果所在位置
    k = len(header_list) - 1

    # result_dict 为分类的结果类型字典
    result_dict = dict(data.iloc[:, k].value_counts())
    # 或使用如下语句：
    # result_dict = dict(data.iloc[:, -1].value_counts())
    result_dict_key = result_dict.keys()

    # 将每个分类结果写入 cate_dict
    # 循环各个分类结果
    for result_key in result_dict_key:
        # 如果类别数据字典不存在该分类结果，默认设置空字典
        if result_key not in cate_dict:
            # dict.setdefault(key, default=None)  键不存在于字典中，将会添加键并将值设为默认值
            cate_dict.setdefault(result_key, {})
        # 在该分类结果下，循环各个条件（因素）
        for factor in factor_list:
            # 如果该分类结果字典不存在该条件（因素），默认设置空字典
            if factor not in cate_dict[result_key]:
                cate_dict[result_key].setdefault(factor, {})
            # 获取该条件的分类列表
            factor_key_list = data[factor].value_counts().index.tolist()
            # 循环获取该条件的各个分类数量
            for key in factor_key_list:
                # 获取该分类结果下，该因素中某个分类的数量
                number = data[(data[header_list[k]] == result_key) & (data[factor] == key)].shape[0]
                if key not in cate_dict[result_key][factor]:
                    cate_dict[result_key][factor].setdefault(key, number)
    return cate_dict, result_dict

def NBeyes(cate_dict, result_dict, new_data):
    # 获取数据集的各个条件（因素）列表
    factor_list = new_data.columns.tolist()

    result_list = []
    # 分类结果列表
    result_key_list = cate_dict.keys()

    # 循环预测新数据
    for i in range(len(new_data)):
        new_result_dict = {}
        # 循环计算各个分类指标的概率
        for result_key in result_key_list:
            # 该分类结果在所有分类结果中的占比
            all_ratio = result_dict[result_key] / sum(list(result_dict.values()))

            # 循环获取该分类结果下，该因素中各个 分类 在 该分类结果 中的占比
            for factor in factor_list:
                ratio = cate_dict[result_key][factor][new_data.iloc[i, factor_list.index(factor)]] / result_dict[
                    result_key]
                # 总占比 乘以 该因素下的各个分类占比
                all_ratio *= ratio
            new_result_dict.setdefault(result_key, all_ratio)

        print(new_result_dict)
        # 获取占比最大的分类结果
        max_result_key = max(new_result_dict, key=new_result_dict.get)
        # 获取占比最大的分类结果的占比
        max_value = new_result_dict[max_result_key]

        result_list.append([max_result_key, max_value])
    return result_list

if __name__ == "__main__":
    #第四题
    X = [59, 59, 64, 66, 66, 66, 68, 68, 68, 69, 70, 71, 72, 74, 75, 75, 75, 76, 78, 80, 81, 81, 81, 82, 82, 82, 83,
         83, 84, 84, 84, 85, 87, 87, 88, 88, 88, 89, 89, 92]

    means, covs, pirors = EM(X, 2, 1000)
    print(means)
    print(covs)
    print(pirors)

    #第五题

    data = pd.read_excel("data.xlsx")
    cate_dict, result_dict = handle_data(data)
    #print(cate_dict)
    #print(result_dict)
    #1.
    new_data_1 = pd.DataFrame({"免费": "T", "抽奖": "T"}, index= [0])
    result1 = NBeyes(cate_dict, result_dict, new_data_1)
    print(result1)
    #2
    new_data_2 = pd.DataFrame({"紧急": "T", "折扣": "T"}, index=[0])
    result2 = NBeyes(cate_dict, result_dict,new_data_2)
    print(result2)