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
