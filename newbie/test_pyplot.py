# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#
# DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False,
#                 sharex=None, sharey=False, layout=None,figsize=None,
#                 use_index=True, title=None, grid=None, legend=True,
#                 style=None, logx=False, logy=False, loglog=False,
#                 xticks=None, yticks=None, xlim=None, ylim=None, rot=None,
#                 xerr=None,secondary_y=False, sort_columns=False, **kwds)
#
# x : label or position, default None#指数据框列的标签或位置参数
# y : label or position, default None
# kind : str
#   ‘line’ : line plot (default)#折线图
#   ‘bar’ : vertical bar plot#条形图
#   ‘barh’ : horizontal bar plot#横向条形图
#   ‘hist’ : histogram#柱状图
#   ‘box’ : boxplot#箱线图
#   ‘kde’ : Kernel Density Estimation plot#Kernel 的密度估计图，主要对柱状图添加Kernel 概率密度线
#   ‘density’ : same as ‘kde’
#   ‘area’ : area plot#不了解此图
#   ‘pie’ : pie plot#饼图
#   ‘scatter’ : scatter plot#散点图  需要传入columns方向的索引
#   ‘hexbin’ : hexbin plot#不了解此图
# title : string#图片的标题用字符串
# grid : boolean, default None (matlab style default)#图片是否有网格
# figsize : 图大小

x=np.linspace(0,2,100)
plt.plot(x,x,label='linear')
# plt.show()

path = '.\\resources\\ex1data1.txt'
data = pd.read_csv(path,names=['Population', 'Profit'])
data.plot(x='Population',y='Profit',kind="scatter",figsize=(12, 8))
plt.show()

# 正弦函数
# 0~2pi 增长0.01
x = np.arange(0,2*np.pi,0.01)
y = np.sin(x)
plt.plot(x,y)
plt.show();

# 正弦函数
# 0~2pi 增长0.01
x = np.arange(0,2*np.pi,0.01)
y = np.cos(x)
plt.plot(x,y)
plt.show();
