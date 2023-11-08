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

