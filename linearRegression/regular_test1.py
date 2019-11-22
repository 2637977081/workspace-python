import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 使用正规方程解决线性回归
path = '.\\resources\\ex1data1.txt'
data = pd.read_csv(path,names=['Population', 'Profit'])


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

plt.scatter(data.Population, data.Profit, label="Training data")
plt.plot(data.Population, data.Population*theta[1] + theta[0], label="Prediction")
plt.legend(loc=2)
plt.show()



