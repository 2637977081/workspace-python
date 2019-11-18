import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 样本文件路径
path = '.\\resources\\ex1data1.txt'
# names添加列名
data = pd.read_csv(path, names=['Population', 'Profit'])
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()


# 获取x
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones,df],axis=1)
    return np.array(data.iloc[:,:-1])


X = get_X(data)
print(X)


# 获取y
def get_y(df):
    return np.array(data.iloc[:,-1])


y = get_y(data)
print(y)


# 缩放特征值
def normalize_feature(df):
    return df.apply(lambda column:(column - column.mean())/column.std())



