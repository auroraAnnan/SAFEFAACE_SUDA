import numpy as np

# 指定要打开的.npy文件的路径
file_path = "E:\Python datas\SAFEFAACE_SUDA\code\model_data\mobilenet_names.npy"

# load函数
data = np.load(file_path)

# 打印数据
print(data)
