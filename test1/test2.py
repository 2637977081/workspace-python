import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 样本文件路径
path = '.\\resources\\ex1data1.txt'
# names添加列名
data = pd.read_csv(path, names=['Population', 'Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


def get_X(df):
    #  读取特征
    #  use concat to add intersect feature to avoid side effect not efficient for big dataset though
    # 使用concat添加intersect特性以避免对大数据集无效的副作用
    #
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()  # 这个操作返回 ndarray,不是矩阵

