# import packages
import numpy as np
import torch

print("创建一个 3*4 的 tensor 矩阵")
matrix = torch.rand(3,4)
print(matrix)
print("matrix's size is :", matrix.size(), "\n")

print("matrix.shape 也可以获取tensor matrix 的尺寸!")
print("matrix's size is :", matrix.shape, "\n")

print("你可以直接从数据中创建tensor")
print("比如： A List of Object")
print("[3, 4.5]")
list = [3, 4.5]
matrix = torch.tensor(list)
print(matrix)
print("matrix's size is :", matrix.shape, "\n")


print("我们也可以对tensor进行计算操作")
A = torch.rand(3,4)
B = torch.rand(3,4)
print("A + B:")
print(A + B, "\n")

print("A - B:")
print(A - B, "\n")

print("torch.add(A, B), 这个代码也可以让两个tensor矩阵相加\m")

print("result 矩阵 是一个用来承接输出的矩阵， 尺寸根据输入矩阵的尺寸定制的")
print("result = torch.empty(A.shape)")
result = torch.empty(A.shape)
torch.add(A, B, out=result)
print("result 矩阵结果：")
print(result)