import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 样本文件路径
path = '.\\resources\\ex1data1.txt'
# names添加列名
data = pd.read_csv(path, names=['Population', 'Profit'])
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
# plt.show()

print("##########读取特征###########")
# 创建全为1的一列
ones = pd.DataFrame({'ones': np.ones(len(data))})
print("ones:%s" % ones)
# 拼接到样本数据中
data = pd.concat([ones, data], axis=1)
print('data:%s ' % data)

print("##########读取标签###########")
print("最后一列[:-1]:%s" % data.iloc[:, -1])
print(np.array(data.iloc[:, -1]))



#
# # 计算代价函数J（θ）
# # 参数theta为特征
# # x中的每行行为x1,x2,x3
# def computeCost(X, y, theta):
#     inner = np.power(((X * theta.T) - y), 2)
#     return np.sum(inner) / (2 * len(X))
#
# # 在训练集的左侧插入一列全为“1”的列，以便计算Cost Function和执行梯度下降
# data.insert(0, 'Ones', 1)
#
# # set X(training set), y(target variable)
# # 设置训练集X，和目标变量y的值
# # 获取列数
# cols = data.shape[1]
# # 输入向量X为前cols-1列
# X = data.iloc[:,0:cols-1]
# # 目标变量y为最后一列
# y = data.iloc[:,cols-1:cols]
#
# X = np.array(X.values)
# y = np.array(y.values)
# theta = np.array([0,0]).reshape(1,2)
#
# print(theta.shape)
# print(X.shape)
# print(y.shape)
# # 参数的数量
# parameters = int(theta.flatten().shape[0])
#
#
# def gradientDescent(X, y, theta, alpha, epoch=1000):
#     '''return theta, cost'''
#     temp = np.array(np.zeros(theta.shape))  # 初始化一个theta临时矩阵
#     parameters = int(theta.flatten().shape[0])  # 参数theta的数量
#     cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次迭代后的cost
#     m = X.shape[0]  # 样本数量m
#
#     for i in range(epoch):
#         # 利用向量化同步计算theta值
#         # 注意theta是一个行向量
#         temp = theta - (alpha / m) * (X.dot(theta.T) - y).T.dot(X)  # 得出一个theta行向量
#         theta = temp
#         cost[i] = computeCost(X, y, theta)  # 这个函数中，theta是变量，X，y是已知量
#     return theta, cost  # 迭代结束之后返回theta和cost值
#
# #初始化学习速率和迭代次数
# alpha = 0.01
# epoch = 2000
#
# #运行梯度下降算法，得出theta和cost
# final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)