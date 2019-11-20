import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")

# 使用正规方程解决线性回归
path = '.\\resources\\ex1data2.txt'
data = pd.read_csv(path, names=['square', 'bedrooms', 'price'])


# print(data)


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


data = normalize_feature(data)

def get_X(df):
    ones = pd.DataFrame({"ones":np.ones(df.shape[0])})
    data = pd.concat([ones,df],axis=1)
    return np.array(data.iloc[:,:-1])


X = get_X(data)


def get_y(df):
    return np.array(data.iloc[:,-1])


y = get_y(data)


def regular(X,y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


theta  = regular(X,y)
print(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.square,data.bedrooms,data.price)
ax.plot_trisurf(data.square,data.bedrooms, theta[0]+theta[1]*data.square+theta[2]*data.bedrooms)
plt.legend(loc = 2)
plt.show()



