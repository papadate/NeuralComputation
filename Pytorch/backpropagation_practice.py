import torch

list = []
# 初始化一个全是1的tensor矩阵， 需要梯度追踪
matrix = torch.ones(4, 1, requires_grad=True)

print(matrix, '\n')
# 修改该tensor矩阵的内容，同时需要重新设置梯度追踪为True
matrix = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
list.append(matrix)
print(matrix)


# 计算过程
matrix_1 = matrix + 2
list.append(matrix_1)
matrix_1.retain_grad()
print(matrix_1)

matrix_2 = matrix_1 * matrix_1 * 3
list.append(matrix_2)
matrix_2.retain_grad()
print(matrix_2)

result = matrix_2.mean()
list.append(result)
result.retain_grad()
print(result, '\n')

# 目前计算节点list 长度
len = len(list)

# 反向计算梯度
result.backward()
for i in range(len):
    print("计算步骤{}: {}".format(i+1, list[i].grad_fn))
print()

for i in range(len):
    index = len - i
    print("梯度{}: {}".format(index, list[index-1].grad))



