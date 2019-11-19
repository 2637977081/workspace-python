import numpy as np

p=np.pi
print(p)

#表示矩阵
# m1 = np.matrix('1,2;3,4')
m1 = np.array([[1, 2, 3], [4, 5, 6],[7,8,9],[10,11,12]])
print("m1:%s" %(m1))

print("---------尺寸----------------")
print("m1 行数: %s" %len(m1))
print("m1 列数: %s" %len(m1[0]))
print("m1 大小: %s" %m1.size)
print("m1 维度(tuple元组):")
print(m1.shape)
print("m1 行数：%s" %m1.shape[0])
print("m1 列数：%s" %m1.shape[1])
print("---------索引截取----------------")
print("m1 截取[1~2)行:")
print(m1[1:2])
print("m1 截取[1~2)列:")
print(m1[:,1:2])
print("---------条件截取----------------")
print("m1 >7:")
print(m1[m1>7])
print("m1 >7 全局布尔判断")
print(m1>7)
mTemp=m1
mTemp[mTemp>7]=0
print("m1 >7 全局判断更新值")
print(mTemp)
print("--------矩阵转置----------------")
print(m1.transpose())
print(m1.T)
print("---------求对数------------------")
print(np.log(m1))
print('-----------有自然数𝑒的幂次运算---------')
print(np.exp(m1))
print('----------反运算-------------')
print(-m1)
print('----------简单数运算-------------')
print(m1+1)

m2 = np.array([[1, 2], [3, 4]])
print("m2:%s" %(m2))


m3 = np.array([[5, 6], [7, 8]])
print("m3:%s" %(m3))

print("矩阵横向合并 :" )
print(np.hstack([m2,m3]))
print(np.concatenate((m2,m3),axis=1))
print("矩阵纵向合并 :" )
print(np.vstack((m2,m3)))
print(np.concatenate((m2,m3),axis=0))

m1=np.array([[1,2,3],[4,5,6]])
m2=np.array([[4,5,6],[1,2,3]])
print("m1+m2:%s" %(m1+m2))

m1=np.array([[1,2,3],[4,5,6]])
m2=np.array([[4,5,6],[1,2,3]])
print("m1-m2:%s" %(m1-m2))

m1=np.array([[1,2,3],[4,5,6]])
m2=np.array([[4,5,6],[1,2,3]])
print("m1*m2:%s" %(m1*m2))

m1=np.array([[1,2,3],[4,5,6]])
m2=np.array([[4,5,6],[1,2,3]])
print("m1/m2:%s" %(m1/m2))

m1=np.array([[1,2,3],[4,5,6]])
m2=np.array([[4,5],[6,1],[2,3]])
print(m1.shape[1]==m2.shape[0])#矩阵点乘要求，列=行
print("m1*m2 点乘:%s" %(np.dot(m1,m2)))
print("m1*m2 点乘:%s" % (m1 @ m2))


m1=np.array([[1,2,3],[4,5,6]])
m2=np.array([[1,2],[3,4],[5,6]])
m3 = np.dot(m1, m2)
print("m3:%s" %(m3))

w1 = np.random.randint(2,size=(2, 3))      #产生2行3列的0,1 int型矩阵
print("w1:%s" %w1)

w2 = np.random.randint(5, size=8)       #产生1行8列的0,1,2,3,4,5 int矩阵
print("w2:%s" %w2)

w3 = np.random.randint(1,4, size=(3,4))       #产生1行8列的0,1,2,3,4,5 int矩阵
print("w3:%s" %w3)

w4 = np.random.rand(3,4)       #产生3行4列的0~1,矩阵
print("w4:%s" %w4)

print('------------linspace 等比/线性-----------')
#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
# start:数据开始点
# stop:数据结束点
# num:样本数据量，默认为50
# endpoint：True则包含stop；False则不包含stop
# retstep：默认为false
# dtype：输出数组类型
# axis：0(默认)或-1
x=np.linspace(0,2,10)
print(x)
x=np.linspace(0,2,10,True)
print(x)
x=np.linspace(0,2,10,False)
print(x)

m = np.array([[1,2],[3,4],[5,6]])
print('------------基础函数-----------:%s' %m)
print("最大值:%s" %m.max())
print("列最大值:%s" %m.max(axis=0))
print("行最大值:%s" %m.max(axis=1))
print("最大值位置:%s" %m.argmax())

print("最小值:%s" %m.min())
print("平均值:%s" %m.mean())
print("方差（相当于【mean(abs(m - m.mean())**2)】）:%s" %m.var())
print("标准差（相当于【sqrt(mean(abs(m - m.mean())**2))】活【sqrt(m.var())】）:%s" %m.std())
print("中值:%s" %np.median(m))
print("求和:%s" %m.sum())
print("累积和:%s" %m.cumsum())

