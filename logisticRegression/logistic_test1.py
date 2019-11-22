import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")

path = '.\\resources\\ex2data1.txt'
data = pd.read_csv(path,names=['param1','param2','admitted'])
print(data)

#获取过滤数据
positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['param1'], positive['param2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['param1'], negative['param2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('param1 Score')
ax.set_ylabel('param2 Score')
plt.show()


# 逻辑函数
def sigmoid(z):
    1/(1+np.exp(-z))


data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros()


# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(X.shape[1])

t =  sigmoid(X * theta.T)

print(t)



def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))



theta = cost(theta,X,y)
print('-代价计算--')
print(theta)


def gradient(theta, X, y):
    print('批量梯度下降-----------------------')
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # R(m,n).T @
    return inner / m

