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

print("result = torch.empty(A.shape)")
print("result 矩阵 是一个用来承接输出的矩阵， 尺寸根据输入矩阵的尺寸定制的")
result = torch.empty(A.shape)
torch.add(A, B, out=result)
print("result 矩阵结果：")
print(result)

print("matrix = torch.ones(5)")
print("torch.ones('row, col' or 'col') 用来创建一个全是1的tensor矩阵")
matrix = torch.ones(5)
print(matrix)
print(matrix.dtype, '\n')

print("目前matrix 是一个 tensor (torch内) 矩阵，\n"
      "他可以通过 matrix.numpy() 转换回numpy 矩阵")
numpy_matrix = matrix.numpy()
print(numpy_matrix)
print(numpy_matrix.dtype)

print("numpy 矩阵 也可以通过 torch.from_numpy() 来转换成torch tensor矩阵")
numpy_matrix = np.ones(10)
print(numpy_matrix)
print(numpy_matrix.dtype, "\n")

print("使用 torch.from_numpy(numpy_matrix)")
torch_matrix = torch.from_numpy(numpy_matrix)
print(torch_matrix)
print(torch_matrix.dtype, "\n")

print("注意：通过from_numpy() 转换过来的tensor矩阵，\n"
      "元素值会随着numpy原矩阵的变化而变化")
print("目前numpy_matrix:")
print(numpy_matrix)
print("随着numpy_matrix转换的tensor矩阵：")
print(torch_matrix,'\n')

print("方法1: np.add(numpy_matrix, 1, out=numpy_matrix)。 不影响torch的 tensor矩阵")
print("因为 xxx = np.add(xxx, 1) 是创建了一个新的矩阵，并不是以前的矩阵")
print("如果我改变num[y_matrix, 每个元素+1")
np.add(numpy_matrix, 1, out=numpy_matrix)
print(numpy_matrix)
print("此时，torch的tensor矩阵也会自己改变：")
print(torch_matrix)

print("方法2: numpy_matrix = np.add(numpy_matrix, 1)。 不影响torch的 tensor矩阵")
print("因为 xxx = np.add(xxx, 1) 是创建了一个新的矩阵，并不是以前的矩阵")
print("如果我改变num[y_matrix, 每个元素+1")
numpy_matrix = np.add(numpy_matrix, 1)
print(numpy_matrix)
print("此时，torch的tensor矩阵不会自己改变：")
print(torch_matrix, '\n')

print("甚至此后的任何改动，发生在numpy_matrix上\n"
      "都不会给tensor矩阵产生任何效果，\n"
      "因为此时的numpy_matrix，已经不是之前转换tensor矩阵的那块内存区域了")
print("你需要重制torch.from_numpy()，来重新更新该tensor矩阵的值")
torch_matrix = torch.from_numpy(numpy_matrix)
print("此时，torch_matrix的值就更新成功了！")
print(torch_matrix,"\n")

print("使用 numpy_matrix.copy()")
print("可以只让tensor获取当前的numpy矩阵的值，后续更新不参与")
numpy_matrix = np.ones((10, 10))
print(numpy_matrix)
torch_matrix = torch.from_numpy(numpy_matrix.copy())
print(torch_matrix)
print("开始更新numpy_matrix")
np.add(numpy_matrix, 1, out=numpy_matrix)
print(numpy_matrix)
print(torch_matrix, '\n')
