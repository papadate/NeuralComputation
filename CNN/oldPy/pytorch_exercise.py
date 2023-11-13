import torch
import numpy as np

matrix_np = np.random.random((3, 4))
print(matrix_np.shape)
matrix_torch = torch.rand(3, 4)
print(matrix_torch.size())
x = torch.tensor([1, 2, 3])
# size() 和 shape 是一个效果。 numpy 里面也是 .shape
print(x.size())
print(x.shape)

# 两个同型torch 矩阵 可以相加
A = torch.rand(3, 4)
B = torch.rand(3, 4)
print(torch.add(A, B))

out = torch.empty(3, 4)
torch.add(A, B, out=out)
print(out)

# torch matrix 可以被转化成 numpy matrix
torch_matrix = torch.ones(5)
print(torch_matrix.dtype)

conversion_matrix = torch_matrix.numpy()
print(conversion_matrix)
print(conversion_matrix.dtype)

np2 = np.ones(3)
torch2 = torch.from_numpy(np2)
print("目前np2 是：\n{}".format(np2))
print("目前torch2 是：\n{}".format(torch2))

np.add(np2, 1, out=np2)
print("np2 整体加1")
print("加1后 np2 是：\n{}".format(np2))
print("加1后 torch2 是：\n{}".format(torch2))
print("所以原来numpy的矩阵被更改后，变化出来的torch矩阵 也会自动更改数值")