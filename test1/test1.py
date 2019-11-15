import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")

# 样本文件路径
path = '.\\resources\\ex1data1.txt'
# names添加列名
data = pd.read_csv(path, names=['Population', 'Profit'])
print("数据样本")
print(data)


# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()


def get_X(df):
    print("##########读取特征###########")
    # 创建全为1的一列
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    # 拼接到样本数据中,根据列合并
    data = pd.concat([ones, df], axis=1)
    # 返回np.array
    print("----------X 去除最后一列[:, :-1]（最后一列是y）---------------")
    print(data.iloc[:, :-1])
    print("数组化")
    print(np.array(data.iloc[:, : -1]))
    return np.array(data.iloc[:, : -1])


X = get_X(data)


def get_y(df):
    print("##########y 读取标签###########")
    print(data.iloc[:, -1])
    print("---------------------")
    print(np.array(data.iloc[:, -1]))
    return np.array(data.iloc[:, -1])


y = get_y(data)


def normalize_feature(df):
    print("--------特征值缩放 X(n)=(X(n)-U(n))/S(n)---------")
    return df.apply(lambda column: (column - column.mean()) / column.std())


print('X-----------')
print(X.shape, type(X))
print('y--------------------')
print(y.shape, type(y))

theta = np.zeros(X.shape[1])
print('-------theta 初始值 X.shape[1]=2,代表特征数n----------')
print(theta)

# print('对比')
# print(X @ theta)
# print('对比2')
# print(theta @ X)

print('X @ theta -------------------')
print(X @ theta)
print('X @ theta - y -------------------')
print(X @ theta - y)


def lr_cost(theta, X, y):
    print('-------代价函数J(theta)计算-------')
    #     """
    #     X: R(m*n), m 样本数, n 特征数
    #     y: R(m)
    #     theta : R(n), 线性回归的参数
    #     """
    m = X.shape[0]  # 样本数
    inner = X @ theta - y  # R(m*1)
    square_sum = inner.T @ inner  # 平方和
    cost = square_sum / (2 * m)
    return cost


theadNum = lr_cost(theta, X, y)
print('代价函数：%s' % theadNum)


def gradient(theta, X, y):
    print('批量梯度下降-----------------------')
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # R(m,n).T @
    return inner / m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    print('拟合线性回归，返回参数和代价---------')
    #     epoch: 批处理的轮数
    #     """
    cost_data = [lr_cost(theta, X, y)]  # 存储变化的特征函数
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆
    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data


epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)
print('final_theta:%s' % final_theta)
print('cost_data: -----------')
print(cost_data)

final_cost = lr_cost(final_theta, X, y)
print('最终代价：%s' % final_cost)
plt.plot(cost_data)
plt.show();


b = final_theta[0]  # intercept，Y轴上的截距
m = final_theta[1]  # slope，斜率

plt.scatter(data.Population, data.Profit, label="Training data")
plt.plot(data.Population, data.Population*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()
