import torch

# 初始化一个全是1的tensor矩阵， 需要梯度追踪
matrix = torch.ones(4, 1, requires_grad=True)
print(matrix, '\n')
# 修改该tensor矩阵的内容，同时需要重新设置梯度追踪为True
matrix = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
print(matrix)


#计算过程
matrix_1 = matrix + 2
# matrix_1.retain_grad()
print(matrix_1)

