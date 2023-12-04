import torch_practice
import torch
import numpy as np

print("初始化 device参数(用来记录GPU是否可用的参数)")
device = None
print(device,'\n')
print("Mac 系统的设备检测过程有些不一样")

print("如果torch.backends.mps.is_available()检测为True")
print("设置 device = 'mps")
print("如果torch.backends.mps.is_available()检测为False")
print("设置 device = 'cpu, 因为你的电脑没有可用GPU")
if (torch.backends.mps.is_available()):
    device = 'mps'
else:
    device = 'cpu'
print("检测后：", device, '\n')

print("当我们发现设备GPU可用后，我们要把torch的tensor矩阵挪到GPU上计算")
print("方法如下：\n")

# 我们使用GPU运算必须在支持GPU的情况下哈！ 用if分开
if(torch.backends.mps.is_available()):
    # 把torch_practice的变量数值，拿过来
    print("我们用 .to('mps')， 设置计算将在GPU上发生")
    print("用GPU计算之前一定要确保dtype是 float32 的")
    print("numpy 可以通过 astype(np.float32)\n"
          "torch 可以通过 .float() 把float64 转换成 float32")
    numpy_matrix = torch_practice.numpy_matrix.astype(np.float32)
    torch_matrix_1 = torch_practice.torch_matrix.float().to(device)
    torch_matrix_2 = torch.from_numpy(numpy_matrix.copy()).to(device)
    result = torch.add(torch_matrix_1, torch_matrix_2)
    print(result)