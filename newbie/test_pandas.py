import pandas as pd

data = [['自有房', 40, 50000],
        ['无自有房', 22, 13000],
        ['自有房', 30, 30000],
        ['自有房', 70, 80000]]
# 格式化数据 index行名，columns列名
# self, data=None, index=None, columns=None, dtype=None, copy=False
# data 矩阵数据
# index 行名
# columns 列名
# dtype 数据格式
data = pd.DataFrame(data, index=['1', '2', '3', '4'], columns=['house', 'age', 'income'])
print("data：%s" % data)
print("0行1列：%s" % data.iloc[0, 1])  # 0行1列
print("1~3行，2~4列：%s" % data.iloc[1:3, 1:2])  # 1~3行，2~4列
print("数据表信息:%s" % data.info())
print("按照value age排序:%s" % data.sort_values(by=['age']))

path = '.\\resources\\ex1data1.txt'
# 读取CSV（逗号分割）文件、文本类型的文件text、log类型到DataFrame
# sep分析器默认','
# delimiter 定界符，备选分隔符（如果指定该参数，则sep参数失效）
# header 指定行数用来作为列名，数据开始行数，如果明确设定 header = 0 就会替换掉原来存在列名
# names 指定列名
data = pd.read_csv(path, names=['Population', 'Profit'])
print("读取文件获取：%s" % data)

print('------------apply函数---------------')
# apply函数是`pandas`里面所有函数中自由度最高的函数。该函数如下：
# DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
# 该函数最有用的是第一个参数，这个参数是函数，相当于C/C++的函数指针。
# 这个函数需要自己实现，函数的传入参数根据axis来定，比如axis = 1，就会把一行数据作为Series的数据
# 会自动遍历每一行DataFrame的数据，最后将所有结果组合成一个Series数据结构并返回。
print("aaa:%s" %data.apply(lambda column: print(column)))
print("----")
print(data.apply(lambda column: (column - column.mean()) / column.std()))
