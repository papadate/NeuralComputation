import torch

print("学习 autograd！")
str = ("autograd 可以自动记录计算过程。\n"
       "当你使用 .backward(), 将自动计算所有梯度。\n"
       "梯度将被记录在 .grad_fn 属性中.\n")
print(str)

print("matrix = torch.ones(2, 2, requires_grad=True)\n"
      "这行代码用来创建一个tensor，同时可以记录计算过程")
matrix = torch.ones(2, 2, requires_grad=True)
print(matrix)

matrix1 = matrix + 2
print(matrix1)

matrix2 = matrix1 * 2
print(matrix2)

matrix3 = matrix2 + 2
print(matrix3, '\n')

def display_grad_fn(list):
    for i in range(len(list)):
        if i == 0:
            print("初始的 matrix，grad_fn 是 None！")
        print(list[i].grad_fn)
    print()

list = [matrix, matrix1, matrix2, matrix3]
display_grad_fn(list)

matrix4 = matrix2 * matrix2 * 2
print(matrix4)

result_scalar = matrix4.mean()
print(result_scalar, '\n')

list.append(matrix4)
list.append(result_scalar)
print("在梯度追踪的计算图里的任何更新操作，都会被记录！\n"
      "即使你从之前的节点开一个小分支去计算新的一条线路。\n")
display_grad_fn(list)

matrix.requires_grad_(False)
print("matrix.requires_grad_(False)")
print("这个操作，将会把matrix设置为不记录计算更新")
print("此后的任何新的计算都没有梯度追踪")

matrix_test = matrix * matrix
print()
print(matrix_test)
print("^^ 此时，你会发现计算出的新tensor矩阵没有梯度追踪记录了！")

print()