import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\machine learning\ex1data1.txt'
data = pd.read_csv(path,names=['Population', 'Profit'])
data.plot(kind='scatter',x='Population', y='Profit', figsize=(12,8))
plt.show()

#参数theta为特征
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
#在训练集的左侧插入一列全为“1”的列，以便计算Cost Function和执行梯度下降
data.insert(0, 'Ones', 1)

