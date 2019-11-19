import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")

# 样本文件路径
path = '.\\resources\\ex1data2.txt'
# names添加列名
data = pd.read_csv(path, names=['square', 'bedrooms', 'price'])


# print(data)


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


data = normalize_feature(data)


def get_X(df):
    ones = pd.DataFrame({"ones": np.ones(df.shape[0])})
    data = pd.concat([ones, df], axis=1)
    return np.array(data.iloc[:, :-1])


X = get_X(data)
print(X)


def get_y(df):
    return np.array(data.iloc[:, -1])


y = get_y(data)
print(y)

theta = np.zeros(data.shape[1])


def lr_cost(theta, X, y):
    m = X.shape[0]
    inner = X @ theta - y
    square_sum = inner.T @ inner
    return square_sum / m


def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()

    for _ in range(epoch):
        _theta = _theta - gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data


epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

theta0 = final_theta[0]
theta1 = final_theta[1]
theta2 = final_theta[2]

plt.plot(cost_data)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.square,data.bedrooms,data.price)
ax.plot_trisurf(data.square,data.bedrooms, theta0+theta1*data.square+theta2*data.bedrooms)
plt.legend(loc = 2)
plt.show()