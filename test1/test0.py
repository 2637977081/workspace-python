import numpy as np

p=np.pi
print(p)




#表示矩阵
# m1 = np.matrix('1,2;3,4')
m1 = np.array([[1, 2, 3], [4, 5, 6]])
print("m1:%s" %(m1))

m2 = np.array([[1, 2], [3, 4], [5, 6]])
print("m2:%s" %(m2))

#矩阵相乘
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



