import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\machine learning\ex1data1.txt'
data = pd.read_csv(path,names=['Population', 'Profit'])
data.plot(kind='scatter',x='Population', y='Profit', figsize=(12,8))
plt.show()